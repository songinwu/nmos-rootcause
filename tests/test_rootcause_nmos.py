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
    assert nk.K == 40
    assert all(nk.PATTERN_CANDIDATES[p] for p in kb.PATTERNS[1:])
    assert np.allclose(nk.SIGNATURES.sum(axis=1), 1.0)


def test_predicted_unique_counts():
    """표만 보고 한 예측 — 측정 11칸이면 23개, 기존 4칸이면 8개가 다른 원인과 겹치지 않는다."""
    assert sum(nk.predicted_unique().values()) == 23
    assert sum(nk.predicted_unique(group4=True).values()) == 8


def test_center_gate_leak_is_three_way_tie():
    assert top_set("center", counts(gate_leak=40, open=3, vt_high=2)) == {
        "n18_rinse_charge", "n23_gate_ash_charge", "n32_contact_ash_charge"}


def test_near_full_vt_high_points_to_vt_dose():
    assert top_set("near-full", counts(vt_high=60, ion_low=4, open=2)) == {"n16_vt_dose"}


def test_near_full_ss_points_to_forming_gas():
    assert top_set("near-full", counts(ss=40, vt_high=30, open=2)) == {"n37_forming_gas_skip"}


def test_uniform_measurements_are_dropped():
    """측정 결과가 무작위와 구별 안 되면 버리고 모양만 본다 — 가장자리 링 후보 전부 동점."""
    c = counts(**{b: 3 for b in nk.BIN_TYPES})
    assert not nk.informative(c)[0]
    assert top_set("edge-ring", c) == set(nk.PATTERN_CANDIDATES["edge-ring"])
