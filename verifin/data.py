"""Synthetic, seeded finance warehouse in SQLite.

Tables: sales (product-level units x ASP), budget (segment revenue plan),
opex, journal_entries (with injected audit anomalies), _labels (ground truth
for those anomalies, used only by evals). All names are fictional.
"""
from __future__ import annotations

import datetime as dt
import sqlite3

import numpy as np

SCHEMA = """
CREATE TABLE sales(period TEXT, segment TEXT, product TEXT, units INTEGER, asp REAL, revenue REAL);
CREATE TABLE budget(period TEXT, segment TEXT, revenue_plan REAL);
CREATE TABLE opex(period TEXT, cost_center TEXT, account TEXT, amount REAL);
CREATE TABLE journal_entries(je_id TEXT PRIMARY KEY, period TEXT, posted_at TEXT, account TEXT,
                             vendor TEXT, amount REAL, posted_by TEXT, description TEXT);
CREATE TABLE _labels(je_id TEXT, kind TEXT);
"""

# product: (base monthly units, base ASP, monthly unit growth, monthly ASP drift)
SEGMENTS = {
    "Data Center": {"Accelerator-H": (9000, 25000.0, 0.045, 0.004),
                    "Accelerator-L": (14000, 9000.0, 0.010, -0.002),
                    "Networking": (30000, 2500.0, 0.030, 0.001)},
    "Gaming": {"GPU-Pro": (220000, 900.0, 0.004, 0.002),
               "GPU-Core": (400000, 350.0, -0.002, -0.001)},
    "ProViz": {"Workstation": (40000, 2200.0, 0.006, 0.001)},
    "Automotive": {"DriveKit": (25000, 1200.0, 0.020, 0.000)},
}
OPEX = {"R&D": {"Compensation": 9e7, "Compute": 3e7}, "S&M": {"Compensation": 2e7, "Marketing": 1.2e7},
        "G&A": {"Compensation": 1.5e7, "Facilities": 6e6}}
ACCOUNTS = ["Travel", "Software", "Consulting", "Facilities"]
VENDORS = ["Acme Logistics", "Borealis Cloud", "Cobalt Advisors", "Delta Travel", "Ember Facilities",
           "Fjord Software", "Granite Legal", "Helix Data", "Iris Catering", "Juniper Staffing"]
USERS = ["akim", "bchen", "cdiaz", "dpatel", "eolsen"]
APPROVAL_THRESHOLD = 10_000.0
MONTHS = [f"{y}-{m:02d}" for y in (2023, 2024, 2025) for m in range(1, 13)]


def _season(seg: str, month: int) -> float:
    return 1.25 if seg == "Gaming" and month in (10, 11, 12) else 1.0


def build(db_path: str = ":memory:", seed: int = 7) -> sqlite3.Connection:
    rng = np.random.default_rng(seed)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    sales, budget, opex = [], [], []
    for t, p in enumerate(MONTHS):
        m = int(p[5:])
        for seg, prods in SEGMENTS.items():
            plan = 0.0
            for prod, (u0, a0, g, d) in prods.items():
                units = int(u0 * (1 + g) ** t * _season(seg, m) * np.exp(rng.normal(0, 0.05)))
                asp = round(a0 * (1 + d) ** t * np.exp(rng.normal(0, 0.01)), 2)
                sales.append((p, seg, prod, units, asp, round(units * asp, 2)))
                plan += u0 * (1 + 0.9 * g) ** t * _season(seg, m) * a0 * (1 + d) ** t
            budget.append((p, seg, round(plan, 2)))
        for cc, accts in OPEX.items():
            for acct, base in accts.items():
                opex.append((p, cc, acct, round(base * 1.02 ** t * np.exp(rng.normal(0, 0.03)), 2)))
    con.executemany("INSERT INTO sales VALUES(?,?,?,?,?,?)", sales)
    con.executemany("INSERT INTO budget VALUES(?,?,?)", budget)
    con.executemany("INSERT INTO opex VALUES(?,?,?,?)", opex)
    _journal(con, rng)
    con.commit()
    return con


def _journal(con: sqlite3.Connection, rng: np.random.Generator) -> None:
    rows, labels = [], []

    def ts(day: dt.date, hour: int) -> str:
        return dt.datetime(day.year, day.month, day.day, hour, int(rng.integers(0, 60))).isoformat()

    for p in MONTHS[-12:]:
        y, m = int(p[:4]), int(p[5:])
        for _ in range(250):
            day = dt.date(y, m, int(rng.integers(1, 29)))
            while day.weekday() >= 5:
                day += dt.timedelta(days=1)
            acct = ACCOUNTS[int(rng.integers(len(ACCOUNTS)))]
            rows.append([day, ts(day, int(rng.integers(8, 19))), acct, VENDORS[int(rng.integers(len(VENDORS)))],
                         round(float(rng.lognormal(7, 1.3)), 2), USERS[int(rng.integers(len(USERS)))],
                         f"{acct} expense"])
    # --- injected anomalies (ground truth) ---
    for i in rng.choice(len(rows), 12, replace=False):          # duplicate payments
        r = list(rows[i]); r[0] = r[0] + dt.timedelta(days=int(rng.integers(0, 3)))
        r[1] = ts(r[0], 10); rows.append(r + ["duplicate"])
    for _ in range(15):                                         # split to dodge approval threshold
        day = dt.date(2025, int(rng.integers(1, 13)), int(rng.integers(1, 29)))
        day -= dt.timedelta(days=max(0, day.weekday() - 4))
        rows.append([day, ts(day, 14), "Consulting", "Cobalt Advisors",
                     round(float(rng.uniform(9500, 9999.99)), 2), "fmorgan", "Advisory services", "threshold"])
    for _ in range(10):                                         # off-hours postings
        day = dt.date(2025, int(rng.integers(1, 13)), int(rng.integers(1, 29)))
        while day.weekday() < 5:
            day += dt.timedelta(days=1)
        rows.append([day, ts(day, int(rng.integers(1, 5))), "Software", "Helix Data",
                     round(float(rng.lognormal(7, 1.0)), 2), "eolsen", "Manual adjustment", "off_hours"])
    for i in rng.choice(250 * 12, 3, replace=False):            # extreme amounts
        r = list(rows[i]); r[4] = round(r[4] * 400, 2); rows[i] = r + ["outlier"]
    order = rng.permutation(len(rows))
    out = []
    for n, idx in enumerate(order, 1):
        r = rows[idx]
        jid = f"JE{n:05d}"
        out.append((jid, r[0].strftime("%Y-%m"), r[1], r[2], r[3], r[4], r[5], r[6]))
        if len(r) > 7:
            labels.append((jid, r[7]))
    con.executemany("INSERT INTO journal_entries VALUES(?,?,?,?,?,?,?,?)", out)
    con.executemany("INSERT INTO _labels VALUES(?,?)", labels)
