import math
import sqlite3

import numpy as np
import pytest

from verifin import tools as T
from verifin.data import build
from verifin.provenance import Ledger


@pytest.fixture(scope="module")
def con():
    return build()


def test_periods():
    assert T.months_in("2025-Q2") == ["2025-04", "2025-05", "2025-06"]
    assert T.prior_period("2025-Q1") == "2024-Q4" and T.prior_period("2025-01") == "2024-12"


def test_pvm_bridge_sums_to_delta(con):
    L = Ledger()
    for seg in (None, "Data Center", "Gaming"):
        f = T.revenue_variance(con, L, "2024", "2025", seg)["facts"]
        assert math.isclose(L[f["price"]].value + L[f["volume"]].value + L[f["mix"]].value, L[f["delta"]].value, rel_tol=1e-9)


def test_sql_guard_blocks_writes_and_labels(con):
    L = Ledger()
    assert T.run_sql(con, L, "SELECT COUNT(*) AS n FROM sales")["rows"][0][0] > 0
    for bad in ("DELETE FROM sales", "DROP TABLE sales", "SELECT * FROM _labels", "ATTACH 'x.db' AS x"):
        with pytest.raises(sqlite3.DatabaseError):
            T.run_sql(con, L, bad)


def test_forecast_intervals_bracket_point(con):
    L = Ledger()
    r = T.forecast(con, L, "Data Center", 3)
    for p in r["points"]:
        assert L[p["lo"]].value < L[p["point"]].value < L[p["hi"]].value
    assert L[r["facts"]["mape_model"]].value < L[r["facts"]["mape_naive"]].value


def test_benford_on_lognormal_conforms():
    amounts = np.random.default_rng(0).lognormal(7, 1.5, 20000)
    assert T.benford(amounts)["verdict"] in ("close conformity", "acceptable conformity")
    assert T.benford(np.full(1000, 55.0))["verdict"] == "nonconformity"


def test_anomaly_recall(con):
    flagged = set(T.journal_anomalies(con, Ledger(), top=0)["flagged_ids"])
    labels = {j for (j,) in con.execute("SELECT je_id FROM _labels")}
    assert len(flagged & labels) / len(labels) >= 0.9
