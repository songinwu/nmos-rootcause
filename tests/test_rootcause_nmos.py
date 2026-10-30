"""NMOS 지식표 (rootcause_kb_nmos) — 표 정합성과 대표 판정."""
import numpy as np

from src.inference import rootcause_kb as kb
from src.inference import rootcause_kb_nmos as nk


def onehot(pattern):
    p = np.zeros(len(kb.PATTERNS))
    p[kb.PATTERNS.index(pattern)] = 1.0
    return p


def counts(**k):
    return np.array([k.get(b, 0) for b in nk.BIN_TYPES], float)


def top_set(pattern, c):
    post = kb.softmax(nk.score(onehot(pattern), c))[0]
    return {nk.CAUSE_IDS[i] for i in np.where(post >= post.max() * (1 - kb.TIE_RTOL))[0]}


def test_table_shape():
    assert nk.K == 39
    assert "n18_rinse_charge" not in nk.CAUSE_INDEX          # 2026-10-30 — 원문(Lin 2012 p.72)과 맞지 않아 뺐다
    assert all(nk.PATTERN_CANDIDATES[p] for p in kb.PATTERNS[1:])
    assert np.allclose(nk.SIGNATURES.sum(axis=1), 1.0)


def test_signature_shares():
    """주 fail 50%(둘이면 25 + 25) · 나머지는 다른 칸에 고르게."""
    s = nk.SIGNATURES[nk.CAUSE_INDEX["n16_vt_dose"]]
    assert np.isclose(s[nk.BIN_TYPES.index("vt_high")], 0.5) and np.isclose(s.min(), 0.5 / (nk.B - 1))
    s = nk.SIGNATURES[nk.CAUSE_INDEX["n01_bevel"]]
    assert np.isclose(s[nk.BIN_TYPES.index("iso_leak")], 0.25) and np.isclose(s[nk.BIN_TYPES.index("metal_leak")], 0.25)
    assert np.isclose(nk.signatures(0.7)[nk.CAUSE_INDEX["n16_vt_dose"]].max(), 0.7)


def test_fail_evidence_counts():
    marks = [nk.fail_evidence(c.id)[0] for c in nk.CAUSES]
    assert (marks.count("✅"), marks.count("🟡"), marks.count("원리")) == (14, 1, 24)


def test_predicted_unique_counts():
    """표만 보고 한 예측 — 측정 12칸이면 21개, 기존 4칸이면 7개가 다른 원인과 겹치지 않는다(2026-10-30)."""
    assert sum(nk.predicted_unique().values()) == 21
    assert sum(nk.predicted_unique(group4=True).values()) == 7


def test_center_gate_leak_is_two_way_tie():
    assert top_set("center", counts(gate_leak=40, open=3, vt_high=2)) == {
        "n23_gate_ash_charge", "n35_metal_ash_charge"}


def test_donut_metal_leak_points_to_develop_residue():
    assert top_set("donut", counts(metal_leak=40, open=3)) == {"n21b_develop_residue"}


def test_new_ties_after_relabel():
    assert top_set("edge-ring", counts(ion_low=40, open=2)) == {"n05_active_focus", "n21a_gate_focus"}
    assert top_set("loc", counts(iso_leak=40, gate_leak=2)) == {"n08_pr_residue", "n10_fill_void"}


def test_near_full_vt_high_points_to_vt_dose():
    assert top_set("near-full", counts(vt_high=60, ion_low=4, open=2)) == {"n16_vt_dose"}


def test_near_full_ss_points_to_forming_gas():
    assert top_set("near-full", counts(ss=40, vt_high=30, open=2)) == {"n37_forming_gas_skip"}


def test_uniform_measurements_are_dropped():
    """측정 결과가 무작위와 구별 안 되면 버리고 모양만 본다 — 가장자리 링 후보 전부 동점."""
    c = counts(**{b: 3 for b in nk.BIN_TYPES})
    assert not nk.informative(c)[0]
    assert top_set("edge-ring", c) == set(nk.PATTERN_CANDIDATES["edge-ring"])
