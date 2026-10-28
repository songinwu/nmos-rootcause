"""NMOS 소자 식 (nmos_device) — 기준값 · 불량 경계 · 「얼마나」 역산."""
import numpy as np
import pytest

from src.inference import nmos_device as nd
from src.inference import rootcause_kb_nmos as nk

FAULTS = {   # 원인 → (파라미터, 사고 크기)
    "n19_gox_thick": ("tox", lambda p: 75e-8),
    "n16_vt_dose": ("Na", lambda p: p["Na"] * 1.45),
    "n20_poly_doping": ("Ndp", lambda p: 5e18),
    "n37_forming_gas_skip": ("Dit", lambda p: p["Dit"] * 100),
    "n24_sd_dose": ("sd", lambda p: p["sd"] * 0.65),            # 2026-10-28 SAB — 막은 S/D 면저항
    "n26_rta_edge": ("dT", lambda p: p["dT"] - 20.0),           # 막은 S/D 면저항 → RTA 온도 (Riley 1996 민감도)
    "n27_salicide_temp": ("c49", lambda p: 0.35),               # silicide 면저항 → 남은 C49 (Clevenger 1994)
}


def test_baseline_matches_sources():
    """기준 — 산화막 60 Å(IDEC), Vt 0.50 V(MOSIS TSMC 0.25 µm 실측)."""
    assert nd.BASE["tox_cv"] == pytest.approx(60.0)
    assert nd.BASE["Vt"] == pytest.approx(0.50, abs=1e-3)
    assert nd.SPEC["Vt"] == pytest.approx((0.40, 0.60))


def test_normal_ranges():
    lo, hi = nd.normal_range("tox")
    assert lo < 60e-8 < hi < 75e-8                 # 75 Å는 불량
    lo, hi = nd.normal_range("Dit")
    assert lo is None and hi < 100 * nd.NOM["Dit"]  # forming gas를 빼먹으면(약 100배) 불량
    lo, hi = nd.normal_range("Ndp")
    assert lo > 5e18                                # poly 도핑 부족은 막은 poly 면저항으로 걸린다
    lo, hi = nd.normal_range("dT")
    assert lo > -20 and hi < 20                     # RTA 가장자리 −20 °C면 막은 S/D 면저항이 합격선 밖


@pytest.mark.parametrize("cause", list(FAULTS))
def test_estimate_recovers_planted_size(cause):
    """90% 범위가 넣은 크기를 대략 90% 담고(웨이퍼 40장), 「하나씩 의심해 보기」에서 그 파라미터 하나만 설명한다.
    한 장만 보면 10%는 범위 밖이 정상이라, 여러 장의 비율로 본다."""
    param, f = FAULTS[cause]
    rng = np.random.default_rng(3)
    inside = only = 0
    for i in range(40):
        p = {k: float(v[0]) for k, v in nd.sample_normal(rng, 1).items()}
        p[param] = float(f(p))
        meas = nd.measure(p, rng)
        e = nd.estimate(cause, meas, n=600, seed=i)
        inside += e["raw"]["p5"] <= p[param] <= e["raw"]["p95"]
        pat = next(c.pattern for c in nk.CAUSES if c.id == cause)   # 맵 모양이 남긴 후보만 — 도즈와 RTA는 모양이 가른다
        only += [r["param"] for r in nd.fingerprint(meas, causes=nk.PATTERN_CANDIDATES[pat]) if r["explains"]] == [param]
    assert inside >= 30 and only >= 34


def test_normal_wafer_explains_nothing():
    rng = np.random.default_rng(5)
    meas = nd.measure({k: float(v[0]) for k, v in nd.sample_normal(rng, 1).items()}, rng)
    assert not any(r["explains"] for r in nd.fingerprint(meas))


def test_no_size_for_margin_cause():
    """여유 유형(ILD 두께 → contact 안 열림)은 칩 측정에 크기가 안 남는다."""
    assert nd.estimate("n29_ild_cmp_edge", nd.BASE) is None
