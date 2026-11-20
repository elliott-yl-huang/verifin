"""Finance tools. Each returns JSON-able data and registers its numbers as facts."""
from __future__ import annotations

import math
import re
import sqlite3
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from .data import APPROVAL_THRESHOLD
from .provenance import Ledger


# ---------------------------------------------------------------- periods
def months_in(period: str) -> List[str]:
    p = period.strip().upper()
    if re.fullmatch(r"\d{4}-\d{2}", p):
        return [p]
    m = re.fullmatch(r"(?:FY)?(\d{4})[\s-]?Q([1-4])", p)
    if m:
        y, q = int(m.group(1)), int(m.group(2))
        return [f"{y}-{mm:02d}" for mm in range(3 * q - 2, 3 * q + 1)]
    m = re.fullmatch(r"(?:FY)?(\d{4})", p)
    if m:
        return [f"{m.group(1)}-{mm:02d}" for mm in range(1, 13)]
    raise ValueError(f"unrecognised period {period!r}; use YYYY, YYYY-Qn or YYYY-MM")


def prior_period(period: str) -> str:
    ms = months_in(period)
    y, m = int(ms[0][:4]), int(ms[0][5:])
    if len(ms) == 12:
        return str(y - 1)
    if len(ms) == 3:
        q = (m - 1) // 3 + 1
        return f"{y - 1}-Q4" if q == 1 else f"{y}-Q{q - 1}"
    return f"{y - 1}-12" if m == 1 else f"{y}-{m - 1:02d}"


def _in(ms: List[str]) -> str:
    return ",".join("?" * len(ms))


# ---------------------------------------------------------------- SQL (guarded)
_ALLOWED = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
            getattr(sqlite3, "SQLITE_RECURSIVE", 33)}


def run_sql(con: sqlite3.Connection, ledger: Ledger, query: str, max_rows: int = 200) -> dict:
    """Read-only SQL enforced by the SQLite authorizer (not by string matching)."""
    def auth(action, arg1, *_):
        if action == sqlite3.SQLITE_READ and arg1 == "_labels":
            return sqlite3.SQLITE_DENY          # eval ground truth is hidden from the agent
        return sqlite3.SQLITE_OK if action in _ALLOWED else sqlite3.SQLITE_DENY

    con.set_authorizer(auth)
    try:
        cur = con.execute(query)
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(max_rows)
    finally:
        con.set_authorizer(lambda *_: sqlite3.SQLITE_OK)  # None only clears on py>=3.11
    facts = {}
    for i, r in enumerate(rows[:50]):
        for c, v in zip(cols, r):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                unit = "usd" if re.search(r"rev|amount|plan|cost|opex|usd|sales", c, re.I) else "count"
                facts[f"{c}[{i}]"] = ledger.add(v, unit, f"sql {c} row {i}", {"tool": "run_sql", "sql": query})
    return {"columns": cols, "rows": rows, "facts": facts}


# ---------------------------------------------------------------- price / volume / mix
def revenue_variance(con, ledger: Ledger, period_a: str, period_b: str, segment: Optional[str] = None) -> dict:
    """Decompose revenue change A->B into price, volume and mix effects (sums exactly to delta)."""
    def agg(period):
        ms = months_in(period)
        q = f"SELECT product, SUM(units), SUM(revenue) FROM sales WHERE period IN ({_in(ms)})"
        args = list(ms)
        if segment:
            q += " AND segment = ?"; args.append(segment)
        return {p: (u, r) for p, u, r in con.execute(q + " GROUP BY product", args)}, q

    a, sql = agg(period_a)
    b, _ = agg(period_b)
    if not a or not b:
        raise ValueError("no sales data for the requested scope")
    prods = sorted(set(a) | set(b))
    ua, ub = sum(a[p][0] for p in a), sum(b[p][0] for p in b)
    ra, rb = sum(a[p][1] for p in a), sum(b[p][1] for p in b)
    avg_asp_a = ra / ua
    price = volume = mix = 0.0
    by_prod = {}
    for p in prods:
        u0, r0 = a.get(p, (0, 0.0)); u1, r1 = b.get(p, (0, 0.0))
        asp0 = r0 / u0 if u0 else r1 / u1
        asp1 = r1 / u1 if u1 else asp0
        pe = (asp1 - asp0) * u1
        mx = asp0 * ub * (u1 / ub - u0 / ua)
        price += pe; mix += mx
        by_prod[p] = r1 - r0
    volume = avg_asp_a * (ub - ua)
    src = {"tool": "revenue_variance", "sql": sql, "period_a": period_a, "period_b": period_b, "segment": segment}
    F = lambda v, u, lbl, inp=(): ledger.add(v, u, lbl, src, inp)
    fa, fb = F(ra, "usd", f"revenue {period_a}"), F(rb, "usd", f"revenue {period_b}")
    fd = F(rb - ra, "usd", "revenue change", (fa, fb))
    out = {"scope": segment or "Total", "facts": {
        "revenue_a": fa, "revenue_b": fb, "delta": fd,
        "delta_pct": F((rb - ra) / ra * 100, "pct", "revenue change %", (fa, fb)),
        "price": F(price, "usd", "price effect"), "volume": F(volume, "usd", "volume effect"),
        "mix": F(mix, "usd", "mix effect")},
        "by_product": {p: F(v, "usd", f"{p} revenue change") for p, v in by_prod.items()}}
    assert math.isclose(price + volume + mix, rb - ra, rel_tol=1e-9, abs_tol=1e-3)
    return out


# ---------------------------------------------------------------- budget vs actual
def budget_variance(con, ledger: Ledger, period: str, segment: Optional[str] = None) -> dict:
    ms = months_in(period)
    seg_sql, args = ("AND segment = ?", [segment]) if segment else ("", [])
    q = (f"SELECT s.segment, s.actual, b.plan FROM (SELECT segment, SUM(revenue) actual FROM sales "
         f"WHERE period IN ({_in(ms)}) {seg_sql} GROUP BY segment) s JOIN (SELECT segment, SUM(revenue_plan) plan "
         f"FROM budget WHERE period IN ({_in(ms)}) {seg_sql} GROUP BY segment) b USING(segment)")
    rows = con.execute(q, ms + args + ms + args).fetchall()
    src = {"tool": "budget_variance", "sql": q, "period": period}
    segs = {}
    for seg, act, plan in rows:
        fa = ledger.add(act, "usd", f"{seg} actual {period}", src)
        fp = ledger.add(plan, "usd", f"{seg} plan {period}", src)
        segs[seg] = {"actual": fa, "plan": fp,
                     "variance": ledger.add(act - plan, "usd", f"{seg} variance", src, (fa, fp)),
                     "variance_pct": ledger.add((act - plan) / plan * 100, "pct", f"{seg} variance %", src, (fa, fp))}
    ta, tp = sum(r[1] for r in rows), sum(r[2] for r in rows)
    fa, fp = ledger.add(ta, "usd", "total actual", src), ledger.add(tp, "usd", "total plan", src)
    return {"segments": segs, "facts": {
        "actual": fa, "plan": fp, "variance": ledger.add(ta - tp, "usd", "total variance", src, (fa, fp)),
        "variance_pct": ledger.add((ta - tp) / tp * 100, "pct", "total variance %", src, (fa, fp))}}


# ---------------------------------------------------------------- forecasting
def _X(t: np.ndarray, month: np.ndarray) -> np.ndarray:
    dummies = (month[:, None] == np.arange(2, 13)[None, :]).astype(float)
    return np.column_stack([np.ones_like(t, dtype=float), t, dummies])


def _fit_predict(y: np.ndarray, months: np.ndarray, t_new: np.ndarray, m_new: np.ndarray, lam=1.0):
    t = np.arange(len(y), dtype=float)
    X = _X(t, months)
    P = lam * np.eye(X.shape[1]); P[0, 0] = 0
    beta = np.linalg.solve(X.T @ X + P, X.T @ np.log(y))
    return _X(t_new.astype(float), m_new) @ beta


def forecast(con, ledger: Ledger, segment: Optional[str] = None, horizon: int = 3, alpha: float = 0.1,
             as_of: Optional[str] = None) -> dict:
    """Log-linear trend + seasonality ridge model, with split-conformal intervals
    calibrated per horizon on rolling-origin backtests; benchmarked vs seasonal naive."""
    horizon = int(max(1, min(horizon, 6)))
    q = "SELECT period, SUM(revenue) FROM sales WHERE period <= ?" + (" AND segment = ?" if segment else "")
    q += " GROUP BY period ORDER BY period"
    data = con.execute(q, [as_of or "9999"] + ([segment] if segment else [])).fetchall()
    periods = [p for p, _ in data]
    y = np.array([v for _, v in data], float)
    mo = np.array([int(p[5:]) for p in periods])
    n = len(y)
    min_train = 12  # one full seasonal cycle; ridge keeps the fit well-posed

    def backtest(lam):
        errs, ape_m, ape_n = {h: [] for h in range(horizon)}, [], []
        for o in range(min_train, n - horizon + 1):
            idx = np.arange(o, o + horizon)
            pred = _fit_predict(y[:o], mo[:o], idx, mo[idx], lam)
            for h, i in enumerate(idx):
                errs[h].append(abs(np.log(y[i]) - pred[h]))
                ape_m.append(abs(np.exp(pred[h]) - y[i]) / y[i])
                ape_n.append(abs(y[i - 12] - y[i]) / y[i])
        return float(np.mean(ape_m)), errs, ape_n

    # choose the ridge penalty by rolling-origin CV (no look-ahead), then calibrate on its residuals
    lam = min((1.0, 0.1, 0.01), key=lambda lm: backtest(lm)[0])
    mape_m, errs, ape_n = backtest(lam)
    t_new = np.arange(n, n + horizon)
    last_y, last_m = int(periods[-1][:4]), int(periods[-1][5:])
    fut = [((last_m - 1 + k) // 12 + last_y, (last_m - 1 + k) % 12 + 1) for k in range(1, horizon + 1)]
    m_new = np.array([m for _, m in fut])
    pred = _fit_predict(y, mo, t_new, m_new, lam)
    src = {"tool": "forecast", "sql": q, "segment": segment, "alpha": alpha, "model": "ridge log-linear + month FE", "lambda": lam}
    points = []
    for h, ((yy, mm), p) in enumerate(zip(fut, pred)):
        e = np.sort(errs[h]); k = len(e)
        qh = e[min(k - 1, math.ceil((k + 1) * (1 - alpha)) - 1)]
        lbl = f"{yy}-{mm:02d}"
        points.append({"period": lbl,
                       "point": ledger.add(math.exp(p), "usd", f"forecast {lbl}", src),
                       "lo": ledger.add(math.exp(p - qh), "usd", f"forecast {lbl} lower", src),
                       "hi": ledger.add(math.exp(p + qh), "usd", f"forecast {lbl} upper", src)})
    return {"scope": segment or "Total", "lambda": lam, "points": points, "facts": {
        "coverage": ledger.add((1 - alpha) * 100, "pct", "nominal interval coverage", src),
        "mape_model": ledger.add(mape_m * 100, "pct", "backtest MAPE model", src),
        "mape_naive": ledger.add(float(np.mean(ape_n)) * 100, "pct", "backtest MAPE seasonal naive", src)}}


# ---------------------------------------------------------------- journal-entry audit
BENFORD = np.log10(1 + 1 / np.arange(1, 10))


def benford(amounts: np.ndarray) -> Dict[str, float]:
    a = amounts[amounts >= 10]
    d = np.array([int(str(int(x))[0]) for x in a])
    obs = np.bincount(d, minlength=10)[1:] / len(d)
    mad = float(np.mean(np.abs(obs - BENFORD)))
    chi2 = float(len(d) * np.sum((obs - BENFORD) ** 2 / BENFORD))
    verdict = ("close conformity" if mad < 0.006 else "acceptable conformity" if mad < 0.012
               else "marginal conformity" if mad < 0.015 else "nonconformity")  # Nigrini thresholds
    return {"mad": mad, "chi2": chi2, "verdict": verdict}


def journal_anomalies(con, ledger: Ledger, period: Optional[str] = None, top: int = 10) -> dict:
    q = "SELECT je_id, period, posted_at, account, vendor, amount, posted_by FROM journal_entries"
    args: list = []
    if period:
        ms = months_in(period); q += f" WHERE period IN ({_in(ms)})"; args = ms
    rows = con.execute(q + " ORDER BY posted_at", args).fetchall()
    import datetime as dt
    flags: Dict[str, List[str]] = defaultdict(list)
    score: Dict[str, float] = defaultdict(float)
    seen: Dict[tuple, List[dt.datetime]] = defaultdict(list)
    thr_users: Dict[str, int] = defaultdict(int)
    logs_by_acct: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        logs_by_acct[r[3]].append(math.log(r[5]))
    stats = {a: (np.median(v), np.median(np.abs(np.array(v) - np.median(v))) or 1e-9) for a, v in logs_by_acct.items()}
    for jid, _, posted, acct, vendor, amt, user in rows:
        t = dt.datetime.fromisoformat(posted)
        key = (vendor, amt)
        if any(abs((t - s).days) <= 5 for s in seen[key]):
            flags[jid].append("possible duplicate payment"); score[jid] += 0.4
        seen[key].append(t)
        if 0.95 * APPROVAL_THRESHOLD <= amt < APPROVAL_THRESHOLD:
            flags[jid].append("just below approval threshold"); score[jid] += 0.3; thr_users[user] += 1
        if t.weekday() >= 5 or t.hour < 6 or t.hour >= 22:
            flags[jid].append("posted off-hours"); score[jid] += 0.25
        med, mad = stats[acct]
        if abs(0.6745 * (math.log(amt) - med) / mad) > 3.5:
            flags[jid].append("extreme amount for account"); score[jid] += 0.35
    for jid, _, _, _, _, _, user in rows:   # repeated threshold-dodging by one user is a pattern
        if "just below approval threshold" in flags.get(jid, []) and thr_users[user] >= 3:
            score[jid] += 0.2; flags[jid].append(f"user {user} has repeated near-threshold entries")
    by_id = {r[0]: r for r in rows}
    ranked = sorted(flags, key=lambda j: (-score[j], j))
    src = {"tool": "journal_anomalies", "sql": q, "period": period}
    b = benford(np.array([r[5] for r in rows]))
    top_rows = [{"je_id": j, "amount": ledger.add(by_id[j][5], "usd", f"{j} amount", src), "vendor": by_id[j][4],
                 "user": by_id[j][6], "risk": round(score[j], 2), "reasons": flags[j]} for j in ranked[:top]]
    return {"flagged_ids": ranked, "top": top_rows, "benford_verdict": b["verdict"], "facts": {
        "n_entries": ledger.add(len(rows), "count", "entries reviewed", src),
        "n_flagged": ledger.add(len(ranked), "count", "entries flagged", src),
        "benford_mad": ledger.add(round(b["mad"], 4), "ratio", "Benford first-digit MAD", src),
        "threshold": ledger.add(APPROVAL_THRESHOLD, "usd", "approval threshold", {"tool": "policy"})}}


def opex_breakdown(con, ledger: Ledger, period: str) -> dict:
    ms = months_in(period); pa = months_in(prior_period(period))
    q = f"SELECT cost_center, SUM(amount) FROM opex WHERE period IN ({_in(ms)}) GROUP BY cost_center ORDER BY 2 DESC"
    cur = dict(con.execute(q, ms).fetchall()); prev = dict(con.execute(q, pa).fetchall())
    src = {"tool": "opex_breakdown", "sql": q, "period": period}
    out = {}
    for cc, v in cur.items():
        f1 = ledger.add(v, "usd", f"{cc} opex {period}", src)
        f0 = ledger.add(prev[cc], "usd", f"{cc} opex prior", src)
        out[cc] = {"amount": f1, "growth_pct": ledger.add((v / prev[cc] - 1) * 100, "pct", f"{cc} opex growth", src, (f0, f1))}
    tot = ledger.add(sum(cur.values()), "usd", f"total opex {period}", src)
    return {"cost_centers": out, "facts": {"total": tot}}


TOOL_FUNCS = {"run_sql": run_sql, "revenue_variance": revenue_variance, "budget_variance": budget_variance,
              "forecast": forecast, "journal_anomalies": journal_anomalies, "opex_breakdown": opex_breakdown}
