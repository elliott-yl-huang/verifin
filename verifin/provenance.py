"""Numeric provenance: a fact ledger + a rounding-aware claim verifier.

Every number a tool produces is registered as a Fact (value, unit, source,
lineage). Any number that appears in an agent's prose is a *claim*; the
verifier parses claims (``$1.23B``, ``-4.5%``, ``1,234``, ``350 bps``),
infers the precision the writer displayed, and accepts the claim only if a
ledger fact rounds to it. Citations like ``[F12]`` are checked too.
"""
from __future__ import annotations

import ast
import operator
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Fact:
    id: str
    value: float
    unit: str            # usd | pct | count | ratio
    label: str
    source: dict
    inputs: Tuple[str, ...] = ()


class Ledger:
    def __init__(self) -> None:
        self.facts: Dict[str, Fact] = {}

    def add(self, value: float, unit: str, label: str, source: dict, inputs=()) -> str:
        fid = f"F{len(self.facts) + 1}"
        self.facts[fid] = Fact(fid, float(value), unit, label, source, tuple(inputs))
        return fid

    def __getitem__(self, fid: str) -> Fact:
        return self.facts[fid]

    def lineage(self, fid: str) -> List[Fact]:
        seen, out, stack = set(), [], [fid]
        while stack:
            f = self.facts[stack.pop()]
            if f.id not in seen:
                seen.add(f.id); out.append(f); stack.extend(f.inputs)
        return out

    def calc(self, expr: str, unit: str, label: str) -> str:
        """Safely evaluate arithmetic over fact ids, e.g. '(F3 - F2) / F2 * 100'."""
        ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
               ast.Div: operator.truediv, ast.USub: operator.neg, ast.UAdd: operator.pos}
        used: List[str] = []

        def ev(n):
            if isinstance(n, ast.Expression):
                return ev(n.body)
            if isinstance(n, ast.BinOp) and type(n.op) in ops:
                return ops[type(n.op)](ev(n.left), ev(n.right))
            if isinstance(n, ast.UnaryOp) and type(n.op) in ops:
                return ops[type(n.op)](ev(n.operand))
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                return n.value
            if isinstance(n, ast.Name) and n.id in self.facts:
                used.append(n.id); return self.facts[n.id].value
            raise ValueError(f"disallowed expression element: {ast.dump(n)[:60]}")

        val = ev(ast.parse(expr, mode="eval"))
        return self.add(val, unit, label, {"tool": "calc", "expr": expr}, inputs=used)

    def to_json(self) -> dict:
        return {k: asdict(v) for k, v in self.facts.items()}


# ---------------------------------------------------------------- formatting
def fmt_usd(v: float) -> str:
    s, a = ("-" if v < 0 else ""), abs(v)
    for scale, suf, d in ((1e9, "B", 2), (1e6, "M", 2), (1e5, "K", 1)):
        if a >= scale:
            return f"{s}${a / scale:,.{d}f}{suf}"
    return f"{s}${a:,.2f}"


def fmt_pct(v: float) -> str:
    return f"{v:+.1f}%"


def cite(ledger: Ledger, fid: str, kind: Optional[str] = None) -> str:
    f = ledger[fid]
    kind = kind or f.unit
    txt = {"usd": fmt_usd, "pct": fmt_pct}.get(kind, lambda v: f"{v:,.0f}" if float(v).is_integer() else f"{v:,.4f}")(f.value)
    return f"{txt} [{fid}]"


# ---------------------------------------------------------------- verification
_NUM = re.compile(r"""
    (?P<cur>\$)?
    (?<![\w.,/#]) (?<!\d-)
    (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    (?!\d|-\d|\.\d)
    \s?(?P<suf>%|pp\b|bps\b|[KMB](?![A-Za-z])|thousand\b|million\b|billion\b)?
    (?:\s*\[(?P<cite>F\d+)\])?
""", re.X)
_SCALE = {"K": 1e3, "M": 1e6, "B": 1e9, "thousand": 1e3, "million": 1e6, "billion": 1e9}


@dataclass
class Claim:
    text: str
    value: float
    tol: float
    kind: str                     # usd | pct | bare
    span: Tuple[int, int]
    cite: Optional[str] = None
    status: str = "ungrounded"    # grounded | ungrounded | citation_mismatch | uncited
    fact: Optional[str] = None


@dataclass
class Verification:
    claims: List[Claim] = field(default_factory=list)

    @property
    def bad(self) -> List[Claim]:
        return [c for c in self.claims if c.status != "grounded"]

    @property
    def ok(self) -> bool:
        return not self.bad

    @property
    def grounding_rate(self) -> float:
        return 1.0 if not self.claims else 1 - len(self.bad) / len(self.claims)

    def feedback(self) -> str:
        return "\n".join(f"- '{c.text}': {c.status}" for c in self.bad)


def extract_claims(text: str) -> List[Claim]:
    out = []
    for m in _NUM.finditer(text):
        raw, suf = m.group("num"), m.group("suf")
        num = float(raw.replace(",", ""))
        dec = len(raw.split(".")[1]) if "." in raw else 0
        if not m.group("cur") and not suf and dec == 0 and (num <= 12 or 1990 <= num <= 2100):
            continue  # ordinals, small counts in prose ("3 months"), years
        scale, kind = 1.0, "bare"
        if suf in _SCALE:
            scale, kind = _SCALE[suf], "usd"
        elif suf in ("%", "pp"):
            kind = "pct"
        elif suf == "bps":
            scale, kind = 0.01, "pct"
        if m.group("cur"):
            kind = "usd"
        tol = 0.5 * 10 ** (-dec) * scale * (1 + 1e-9) + 1e-9
        out.append(Claim(m.group(0).strip(), num * scale, tol, kind, m.span(), m.group("cite")))
    return out


def _matches(c: Claim, f: Fact) -> bool:
    if c.kind == "pct" and f.unit != "pct":
        return False
    if c.kind == "usd" and f.unit != "usd":
        return False
    return abs(abs(f.value) - c.value) <= c.tol


def verify(text: str, ledger: Ledger, require_citations: bool = True) -> Verification:
    v = Verification(extract_claims(text))
    for c in v.claims:
        if c.cite:
            f = ledger.facts.get(c.cite)
            if f and _matches(c, f):
                c.status, c.fact = "grounded", f.id
            else:
                c.status = "citation_mismatch"
            continue
        hit = next((f for f in ledger.facts.values() if _matches(c, f)), None)
        if hit:
            c.fact = hit.id
            c.status = "uncited" if require_citations else "grounded"
    return v


def redact(text: str, v: Verification) -> str:
    """Fail closed: replace every unverified number with a visible marker."""
    for c in sorted(v.bad, key=lambda c: -c.span[0]):
        text = text[:c.span[0]] + "[UNVERIFIED]" + text[c.span[1]:]
    return text
