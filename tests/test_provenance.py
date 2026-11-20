import pytest

from verifin.provenance import Ledger, extract_claims, fmt_usd, redact, verify


@pytest.fixture
def L():
    L = Ledger()
    L.add(1_234_560_000, "usd", "rev", {})       # F1
    L.add(12.345, "pct", "growth", {})           # F2
    L.add(3037, "count", "n", {})                # F3
    return L


def test_rounding_aware_match(L):
    assert verify("Revenue was $1.23B [F1].", L).ok
    assert verify("Revenue was $1,234.6M [F1].", L).ok          # different display, same fact
    assert not verify("Revenue was $1.25B [F1].", L).ok         # outside rounding tolerance


def test_citation_mismatch_and_uncited(L):
    v = verify("Growth was +12.3% [F1].", L)
    assert v.claims[0].status == "citation_mismatch"
    v = verify("Growth was 12.3%.", L)
    assert v.claims[0].status == "uncited" and v.claims[0].fact == "F2"
    assert verify("Growth was 12.3%.", L, require_citations=False).ok


def test_ignores_dates_ids_and_years(L):
    text = "In 2025-Q3 and 2025-03, entry JE00120 [F3] for fiscal 2024 over 3 months."
    assert [c.text for c in extract_claims(text)] == []


def test_units_and_bps(L):
    L.add(4.12, "pct", "margin", {})             # F4
    assert verify("Margin up 412 bps [F4].", L).ok
    assert not verify("Revenue $12.3 [F2].", L).ok             # usd claim cannot match a pct fact


def test_redact_fail_closed(L):
    text = "Revenue $1.23B [F1], margin 55.0% [F2]."
    v = verify(text, L)
    out = redact(text, v)
    assert "$1.23B" in out and "55.0%" not in out and "[UNVERIFIED]" in out


def test_calc_lineage_and_safety(L):
    fid = L.calc("F1 / F3", "usd", "rev per entry")
    assert {f.id for f in L.lineage(fid)} == {fid, "F1", "F3"}
    with pytest.raises(ValueError):
        L.calc("__import__('os')", "usd", "x")


def test_fmt_usd():
    assert fmt_usd(-695_530_000) == "-$695.53M" and fmt_usd(9950.5) == "$9,950.50"
