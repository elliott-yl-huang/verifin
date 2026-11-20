"""The agent: skill routing -> tool calls -> narrative -> verify -> repair -> redact.

Two planners share the same tools, ledger and verifier:
* LLM planner (OpenAI-compatible tool calling, e.g. NVIDIA NIM) when an API key is set.
* Deterministic offline planner (skill playbooks) for CI, tests and air-gapped use.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from . import tools as T
from .llm import ChatClient
from .provenance import Ledger, Verification, cite, redact, verify
from .skills import load_skills, route, skill_index

SEGMENTS = {"data center": "Data Center", "datacenter": "Data Center", "gaming": "Gaming", "proviz": "ProViz",
            "professional visualization": "ProViz", "automotive": "Automotive"}

TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": n, "description": d, "parameters": {"type": "object", "properties": p, "required": r}}}
    for n, d, p, r in [
        ("load_skill", "Load the full playbook for a skill.", {"name": {"type": "string"}}, ["name"]),
        ("run_sql", "Read-only SQL over the finance warehouse.", {"query": {"type": "string"}}, ["query"]),
        ("revenue_variance", "Price/volume/mix bridge between two periods.",
         {"period_a": {"type": "string"}, "period_b": {"type": "string"}, "segment": {"type": "string"}}, ["period_a", "period_b"]),
        ("budget_variance", "Actual vs plan revenue.", {"period": {"type": "string"}, "segment": {"type": "string"}}, ["period"]),
        ("forecast", "Revenue forecast with conformal intervals.",
         {"segment": {"type": "string"}, "horizon": {"type": "integer"}, "alpha": {"type": "number"}}, []),
        ("journal_anomalies", "Audit screen of journal entries.", {"period": {"type": "string"}, "top": {"type": "integer"}}, []),
        ("opex_breakdown", "Opex by cost center with growth.", {"period": {"type": "string"}}, ["period"]),
        ("calc", "Arithmetic over fact ids, e.g. '(F3-F2)/F2*100'. Returns a new cited fact.",
         {"expr": {"type": "string"}, "unit": {"type": "string", "enum": ["usd", "pct", "count", "ratio"]},
          "label": {"type": "string"}}, ["expr", "unit", "label"]),
    ]]

SYSTEM = """You are a finance analyst agent. Data covers 2023-01..2025-12 (monthly).
Skills available (call load_skill before using one):
{skills}
HARD RULES: Every number you write must come from a tool result and be followed by its fact id in
brackets, e.g. "$1.23B [F4]" or "+5.2% [F9]". Use calc for any derived number. Do not invent numbers.
Write for a finance manager: short, plain business language, lead with the answer."""


@dataclass
class Answer:
    question: str
    skill: str
    text: str
    verification: Verification
    ledger: Ledger
    mode: str
    repairs: int = 0
    trace: List[dict] = field(default_factory=list)

    def audit_pack(self) -> dict:
        used = {c.fact for c in self.verification.claims if c.fact}
        lineage = {f.id: f for fid in used for f in self.ledger.lineage(fid)}
        return {"question": self.question, "skill": self.skill, "mode": self.mode, "answer": self.text,
                "verified": self.verification.ok, "grounding_rate": self.verification.grounding_rate,
                "repairs": self.repairs, "trace": self.trace,
                "claims": [{"text": c.text, "status": c.status, "fact": c.fact} for c in self.verification.claims],
                "facts": {k: self.ledger.to_json()[k] for k in sorted(lineage, key=lambda s: int(s[1:]))}}


# ---------------------------------------------------------------- param parsing (offline)
_PERIOD = re.compile(r"\b(?:FY)?(\d{4})[\s-]?Q([1-4])\b|\bQ([1-4])[\s-]?(?:FY)?(\d{4})\b|\b(\d{4}-\d{2})\b|\b(?:FY)?(\d{4})\b", re.I)


def parse_periods(q: str) -> List[str]:
    out = []
    for m in _PERIOD.finditer(q):
        if m.group(1):
            out.append(f"{m.group(1)}-Q{m.group(2)}")
        elif m.group(3):
            out.append(f"{m.group(4)}-Q{m.group(3)}")
        else:
            out.append(m.group(5) or m.group(6))
    return out


def parse_segment(q: str) -> Optional[str]:
    ql = q.lower()
    return next((v for k, v in SEGMENTS.items() if k in ql), None)


# ---------------------------------------------------------------- offline playbooks
def _pb_variance(con, L, q):
    ps = sorted(parse_periods(q), key=lambda p: T.months_in(p)[0])
    a, b = (ps[0], ps[-1]) if len(ps) >= 2 else ((T.prior_period(ps[0]), ps[0]) if ps else ("2025-Q3", "2025-Q4"))
    seg = parse_segment(q)
    r = T.revenue_variance(con, L, a, b, seg)
    f = r["facts"]
    top = max(r["by_product"], key=lambda p: abs(L[r["by_product"][p]].value))
    direction = "grew" if L[f["delta"]].value >= 0 else "declined"
    return (f"{r['scope']} revenue {direction} from {cite(L, f['revenue_a'])} in {a} to {cite(L, f['revenue_b'])} in {b}, "
            f"a change of {cite(L, f['delta'])} ({cite(L, f['delta_pct'])}).\n"
            f"Bridge: price effect {cite(L, f['price'])}, volume effect {cite(L, f['volume'])}, "
            f"mix effect {cite(L, f['mix'])}.\n"
            f"Largest product driver: {top} at {cite(L, r['by_product'][top])}.")


def _pb_budget(con, L, q):
    ps = parse_periods(q); p = ps[0] if ps else "2025"
    r = T.budget_variance(con, L, p, parse_segment(q))
    f = r["facts"]
    tone = "favorable" if L[f["variance"]].value >= 0 else "unfavorable"
    lines = [f"{p} revenue was {cite(L, f['actual'])} against a plan of {cite(L, f['plan'])}: "
             f"a {tone} variance of {cite(L, f['variance'])} ({cite(L, f['variance_pct'])})."]
    for seg, s in sorted(r["segments"].items(), key=lambda kv: -abs(L[kv[1]["variance"]].value)):
        lines.append(f"- {seg}: actual {cite(L, s['actual'])} vs plan {cite(L, s['plan'])}, "
                     f"variance {cite(L, s['variance'])} ({cite(L, s['variance_pct'])})")
    return "\n".join(lines)


def _pb_forecast(con, L, q):
    m = re.search(r"next (\d+) months?", q, re.I)
    h = int(m.group(1)) if m else 3
    r = T.forecast(con, L, parse_segment(q), h)
    f = r["facts"]
    lines = [f"{r['scope']} revenue forecast ({cite(L, f['coverage'])[1:]} conformal intervals):"]
    for p in r["points"]:
        lines.append(f"- {p['period']}: {cite(L, p['point'])} (range {cite(L, p['lo'])} to {cite(L, p['hi'])})")
    lines.append(f"Backtest MAPE {cite(L, f['mape_model'])[1:]} vs seasonal-naive {cite(L, f['mape_naive'])[1:]}.")
    return "\n".join(lines)


def _pb_audit(con, L, q):
    ps = parse_periods(q)
    r = T.journal_anomalies(con, L, ps[0] if ps else None, top=5)
    f = r["facts"]
    lines = [f"Reviewed {cite(L, f['n_entries'])} journal entries{' in ' + ps[0] if ps else ''}; "
             f"{cite(L, f['n_flagged'])} were flagged. Benford first-digit MAD is {cite(L, f['benford_mad'])} "
             f"({r['benford_verdict']}). Approval threshold: {cite(L, f['threshold'])}.", "Highest-risk entries:"]
    for e in r["top"]:
        lines.append(f"- {e['je_id']} {cite(L, e['amount'])} to {e['vendor']} by {e['user']}: {'; '.join(e['reasons'])}")
    lines.append("Recommended: pull supporting documents and approver sign-off for these entries.")
    return "\n".join(lines)


def _pb_opex(con, L, q):
    ps = parse_periods(q); p = ps[0] if ps else "2025-Q4"
    r = T.opex_breakdown(con, L, p)
    lines = [f"Total opex in {p} was {cite(L, r['facts']['total'])}."]
    for cc, d in r["cost_centers"].items():
        lines.append(f"- {cc}: {cite(L, d['amount'])} ({cite(L, d['growth_pct'])} vs {T.prior_period(p)})")
    return "\n".join(lines)


def _pb_adhoc(con, L, q):
    return "This question needs the LLM planner to write SQL. Set NVIDIA_API_KEY (or VERIFIN_API_KEY) and retry."


PLAYBOOKS: Dict[str, Callable] = {"revenue-variance": _pb_variance, "budget-vs-actual": _pb_budget,
                                  "revenue-forecast": _pb_forecast, "journal-entry-audit": _pb_audit,
                                  "opex-breakdown": _pb_opex, "adhoc-sql": _pb_adhoc}


# ---------------------------------------------------------------- agent
class Agent:
    def __init__(self, con, llm: Optional[ChatClient] = None, max_repairs: int = 2, max_steps: int = 12):
        self.con, self.llm, self.max_repairs, self.max_steps = con, llm, max_repairs, max_steps
        self.skills = load_skills()

    def ask(self, question: str) -> Answer:
        return self._ask_llm(question) if self.llm else self._ask_offline(question)

    def _ask_offline(self, question: str) -> Answer:
        L, skill = Ledger(), route(question, self.skills)
        t0 = time.time()
        text = PLAYBOOKS[skill](self.con, L, question)
        trace = [{"step": "playbook", "skill": skill, "ms": round((time.time() - t0) * 1000, 1)}]
        v = verify(text, L)
        if not v.ok:
            text = redact(text, v)
        return Answer(question, skill, text, v, L, "offline", 0, trace)

    # -- LLM tool-calling loop with verify -> repair -> redact
    def _dispatch(self, L: Ledger, name: str, args: dict):
        if name == "load_skill":
            s = self.skills.get(args.get("name", ""))
            return s.body if s else f"unknown skill; choose from {list(self.skills)}"
        if name == "calc":
            fid = L.calc(args["expr"], args.get("unit", "usd"), args.get("label", "derived"))
            return {"fact": fid, "display": cite(L, fid)}
        return _render(T.TOOL_FUNCS[name](self.con, L, **args), L)

    def _ask_llm(self, question: str) -> Answer:
        L, hint = Ledger(), route(question, self.skills)
        msgs = [{"role": "system", "content": SYSTEM.format(skills=skill_index(self.skills))},
                {"role": "user", "content": f"{question}\n(Router hint: skill '{hint}' looks relevant.)"}]
        trace, repairs, text, v = [], 0, "", verify("", L)
        for _ in range(self.max_steps):
            msg = self.llm.chat(msgs, TOOL_SCHEMAS)
            msgs.append({k: msg[k] for k in ("role", "content", "tool_calls") if msg.get(k) is not None})
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    name = tc["function"]["name"]
                    t0 = time.time()
                    try:
                        args = json.loads(tc["function"].get("arguments") or "{}")
                        out = self._dispatch(L, name, args)
                    except Exception as e:  # tool errors go back to the model, not the user
                        args, out = {}, {"error": str(e)}
                    trace.append({"tool": name, "args": args, "ms": round((time.time() - t0) * 1000, 1)})
                    msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(out, default=str)[:12000]})
                continue
            text = msg.get("content") or ""
            v = verify(text, L)
            if v.ok or repairs >= self.max_repairs:
                break
            repairs += 1
            trace.append({"step": "repair", "problems": len(v.bad)})
            msgs.append({"role": "user", "content": "Verifier rejected these numbers (not traceable to a tool fact, "
                         "or wrong/missing citation). Rewrite the answer fixing them; use calc if needed:\n" + v.feedback()})
        if not v.ok:
            text = redact(text, v)
        return Answer(question, hint, text, v, L, f"llm:{self.llm.model}", repairs, trace)


def _render(obj, L: Ledger):
    """Expand fact ids in tool output into {fact, value, display} so the model can cite them."""
    if isinstance(obj, dict):
        return {k: _render(v, L) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_render(v, L) for v in obj]
    if isinstance(obj, str) and re.fullmatch(r"F\d+", obj) and obj in L.facts:
        return {"fact": obj, "value": round(L[obj].value, 4), "display": cite(L, obj)}
    return obj
