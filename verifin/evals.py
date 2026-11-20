"""Evaluation harness: accuracy vs SQL oracles, grounding, red-team hallucination detection,
anomaly-detector precision/recall, and out-of-sample forecast interval coverage."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from . import tools as T
from .agent import Agent
from .data import build
from .provenance import Ledger, verify

GOLDEN = Path(__file__).resolve().parent.parent / "evals" / "golden.jsonl"
THRESHOLDS = {"routing_acc": 0.9, "answer_acc": 0.9, "grounding": 1.0, "redteam_detection": 1.0,
              "anomaly_recall": 0.9, "forecast_coverage": 0.8}


def _perturb(text: str, claim) -> str:
    """Change one cited number by ~7% while keeping its format and citation (a subtle hallucination)."""
    m = re.search(r"\d[\d,]*(?:\.\d+)?", claim.text)
    raw = m.group(0); dec = len(raw.split(".")[1]) if "." in raw else 0
    new = f"{float(raw.replace(',', '')) * 1.07 + 10 ** -dec:,.{dec}f}"
    s, _ = claim.span
    return text[:s] + claim.text.replace(raw, new, 1) + text[s + len(claim.text):]


def run(golden: Path = GOLDEN, agent: Optional[Agent] = None) -> dict:
    con = build()
    agent = agent or Agent(con)
    rows, ok_route, ok_ans, ground, caught, attacks = [], 0, 0, [], 0, 0
    for line in golden.read_text().splitlines():
        g = json.loads(line)
        a = agent.ask(g["q"])
        oracle = con.execute(g["oracle"]).fetchone()[0]
        hit = any(c.status == "grounded" and abs(abs(oracle) - c.value) <= c.tol for c in a.verification.claims)
        ok_route += a.skill == g["skill"]; ok_ans += hit; ground.append(a.verification.grounding_rate)
        # red team: (1) subtle perturbation of a cited number, (2) an uncited fabricated figure
        good = [c for c in a.verification.claims if c.status == "grounded"]
        for bad_text in ([_perturb(a.text, good[0])] if good else []) + [a.text + " Margin also improved 412 bps."]:
            attacks += 1; caught += not verify(bad_text, a.ledger).ok
        rows.append({"q": g["q"], "skill": a.skill, "route_ok": a.skill == g["skill"], "answer_ok": hit,
                     "grounding": a.verification.grounding_rate})
    n = len(rows)
    # anomaly detector vs ground-truth labels
    L = Ledger()
    flagged = set(T.journal_anomalies(con, L, top=0)["flagged_ids"])
    labels = {j for (j,) in con.execute("SELECT je_id FROM _labels")}
    # out-of-sample conformal coverage: forecast from several origins, score against realised actuals
    inside = total = 0
    for as_of in ("2024-12", "2025-03", "2025-06", "2025-09"):
        for seg in (None, "Data Center", "Gaming", "ProViz", "Automotive"):
            fc = T.forecast(con, L, seg, 3, 0.1, as_of=as_of)
            for p in fc["points"]:
                q = "SELECT SUM(revenue) FROM sales WHERE period = ?" + (" AND segment = ?" if seg else "")
                act = con.execute(q, [p["period"]] + ([seg] if seg else [])).fetchone()[0]
                if act is None:
                    continue
                total += 1; inside += L[p["lo"]].value <= act <= L[p["hi"]].value
    m = {"n": n, "routing_acc": ok_route / n, "answer_acc": ok_ans / n, "grounding": sum(ground) / n,
         "redteam_detection": caught / attacks, "anomaly_precision": len(flagged & labels) / max(1, len(flagged)),
         "anomaly_recall": len(flagged & labels) / len(labels), "forecast_coverage": inside / max(1, total)}
    m["passed"] = all(m[k] >= v for k, v in THRESHOLDS.items())
    return {"metrics": m, "rows": rows}
