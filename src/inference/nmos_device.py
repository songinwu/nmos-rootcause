"""
NMOS 소자 식 — 공정값 ↔ WAT 측정값.
(2026-10-28) 원인 공정 추천의 「얼마나」. 엔진(rootcause_kb_nmos)이 맵 모양 + 칩 측정 칸으로 단계를 고르면, 그 단계의 파라미터가
얼마나 벗어났는지를 웨이퍼 WAT 값에서 거꾸로 계산하고 90% 범위를 같이 낸다. EDS 맵이 「어느 단계」, WAT 값이 「얼마나」를 맡는다.

    공정값      : gate 산화막 두께 · 채널 도핑 · poly 도핑 · 계면 트랩 · S/D 활성 도즈 · RTA 온도 이탈 · 남은 C49 비율
    WAT 값      : Vt · SS · on 전류 · C-V 산화막 · 면저항 셋(silicide를 막은 S/D · 막은 poly · silicide 덮인 곳)
    기준값      : gate 산화막 60 Å (IDEC 0.5 µm 강의) · Vt 0.50 V (MOSIS TSMC 0.25 µm 실측 — 산화막 57 Å, Vt 0.49~0.50 V)
                  → 채널 도핑은 식으로 역산 · poly 10²⁰ cm⁻³ (IDEC 「10²⁰~10²²」의 아래 끝) · 계면 트랩 10¹⁰ cm⁻²eV⁻¹ (가정 —
                  forming gas 뒤) · 면저항 막은 N+ 58 · 막은 poly 176 · silicide 덮인 N+ 3.5 Ω/sq (MOSIS T7CZ 실측)
    측정 합격선 : Vt ±0.10 V · on 전류 ±16% (GF180MCU 3.3 V NMOS 공개 spec의 폭 — 0.53/0.63/0.73 V · 430/510/590 µA/µm)
                  · SS 기준 +10 mV/dec (가정) · 면저항 ±15% (가정 — GF180MCU 금속 면저항 공차 ±15.6%와 같은 폭)
    식          : 장채널 MOS 문턱 전압(poly 일함수 · poly 공핍 · 계면 트랩 포함) · SS · alpha-power on 전류 (Taur & Ning ·
                  Sze, 위키 밖 교과서 지식). 면저항 *(2026-10-28 SAB 추가)* —
                    막은 S/D  = 58 / (활성 도즈 비) − 0.5 Ω/sq × (RTA 온도 이탈 °C)
                                온도 민감도는 Riley 1996 p.20(비소 · 1,025 °C · 2 °C/(Ω/sq))을 옮김 — 주입 조건이 달라 근사
                    막은 poly = 176 × (10²⁰ / poly 도핑)          고농도에서 면저항 ∝ 1/도핑 (원리)
                    silicide  = 3.5 × (1 + 남은 C49 비율 × (65 / 17.5 − 1))   C49 60~70 · C54 15~20 µΩ·cm (Clevenger 1994
                                p.7874)
    불량 경계   : 파라미터 하나만 움직였을 때 측정이 전부 합격선 안에 머무는 범위 (normal_range)

⚠️ 한계
- 같은 식으로 사고를 심고 같은 식으로 역산하면 범위가 좁게 나온다 — 「식이 맞다면」의 상한이다. 실제로는 기준 웨이퍼와의 차이로
  쓰고, 식은 「얼마나 민감한가」에만 쓴다.
- 정상 산포(산화막 σ 1.3% · 도핑 σ 3% · poly σ 3% · 계면 트랩 ×÷1.3 · 남은 C49 1%), 측정 잡음, 전원 2.5 V, alpha 1.3은 가정이다.
  원문에서 온 것은 RTA 온도 σ 1 °C(Riley 1996 p.22 — 장비 3σ ±3 °C)와 S/D 도즈 σ 0.5%(같은 쪽 — 비소 모니터 면저항
  3σ 2.04 / 137 Ω/sq ≈ 1.5%, 주입 · 열처리 · 측정을 합친 산포라 상한으로 씀)다.
- S/D 도즈 · RTA는 on 전류에 거의 안 나타난다 — 0.5 µm면 S/D 직렬 저항(수십 Ω·µm)이 채널 저항(수 kΩ·µm)보다 훨씬 작아서다
  (원리). 그래서 막은 S/D 면저항으로만 잰다. 막은 면저항 하나로는 도즈 부족과 RTA 온도 부족이 같아 보이고, 둘은 맵 모양
  (거의 전체 vs 가장자리 링)이 가른다.
- 소자 값(산화막 몇 Å)까지만 말한다. 장비 설정(산화로 몇 °C · 몇 분)은 같은 두께를 만드는 조합이 여러 개라 정하지 않는다.
  예외는 RTA — 면저항 ↔ 온도 민감도를 공장이 실제로 이렇게 쓴다(Riley 1996).
- 사고가 둘 겹치면 이 역산은 틀린다 — 그래서 다른 측정과 맞는지(consistent)를 같이 본다.

torch를 import하지 않는다.
"""
from __future__ import annotations

import numpy as np

Q, E0 = 1.602e-19, 8.854e-14
ES, EOX = 11.7 * E0, 3.9 * E0
NI, KT, NC, EG2 = 1.0e10, 0.02585, 2.8e19, 0.56
SS0 = 59.6                     # ln10 · kT/q (mV/dec, 300 K)
VDD, ALPHA = 2.5, 1.3          # 가정 — 57~60 Å 세대 전원(TSMC 0.25 µm 2.5 V) · 속도 포화 반영 지수
RS_SD0, RS_POLY0, RS_SIL0 = 58.0, 176.0, 3.5     # Ω/sq — MOSIS T7CZ: N+BLK · PLY+BLK · N+(silicide)
DRS_DT = 0.5                   # Ω/sq per °C — Riley 1996 p.20 (2 °C/(Ω/sq))
RHO_RATIO = 65.0 / 17.5        # C49 / C54 비저항 — Clevenger 1994 p.7874

VT_TARGET = 0.50
NOM = {"tox": 60e-8, "Na": None, "Ndp": 1e20, "Dit": 1e10, "sd": 1.0, "dT": 0.0, "c49": 0.0}
SPREAD = {"tox": 0.013, "Na": 0.03, "Ndp": 0.03, "Dit": 0.26, "sd": 0.005, "dT": 1.0, "c49": 0.01}   # 정상 산포 (Dit는 로그 σ)
NOISE = {"Vt": 0.005, "SS": 1.0, "ion": 0.01, "tox_cv": 0.01,                                    # 웨이퍼 중앙값 측정 잡음
         "rs_sd": 0.005, "rs_poly": 0.01, "rs_sil": 0.01}
REL_NOISE = {"ion", "tox_cv", "rs_sd", "rs_poly", "rs_sil"}       # 잡음이 상대값인 측정
MEAS = ["Vt", "SS", "ion", "tox_cv", "rs_sd", "rs_poly", "rs_sil"]
MEAS_LABEL = {"Vt": "Vt", "SS": "SS", "ion": "on 전류", "tox_cv": "산화막 (C-V)",
              "rs_sd": "S/D 면저항 (막은 패턴)", "rs_poly": "poly 면저항 (막은 패턴)", "rs_sil": "silicide 면저항"}

KNOB = {   # 파라미터 → (화면 이름, 역산에 쓰는 측정)
    "tox": ("gate 산화막 두께", "tox_cv"),
    "Na": ("채널 도핑 (Vt 주입 도즈)", "Vt"),
    "Ndp": ("poly 도핑", "rs_poly"),
    "Dit": ("계면 트랩", "SS"),
    "sd": ("S/D 활성 도즈", "rs_sd"),
    "dT": ("RTA 온도", "rs_sd"),
    "c49": ("남은 고저항 상 (C49)", "rs_sil"),
}
CAUSE_KNOB = {"n16_vt_dose": "Na", "n17_screen_residue": "tox", "n19_gox_thick": "tox",
              "n20_poly_doping": "Ndp", "n37_forming_gas_skip": "Dit",
              "n24_sd_dose": "sd", "n26_rta_edge": "dT", "n27_salicide_temp": "c49"}
_GRID = {"tox": (30e-8, 150e-8, False), "Na": (1e16, 1e19, True), "Ndp": (1e17, 1e21, True), "Dit": (1e8, 1e14, True),
         "sd": (0.2, 3.0, True), "dT": (-150.0, 100.0, False), "c49": (0.0, 1.0, False)}


def device(tox, Na, Ndp, Dit):
    """공정값 → (Vt [V], SS [mV/dec], on 전류 [임의 단위], Cox [F/cm²]). 배열을 받는다."""
    tox, Na, Ndp, Dit = np.broadcast_arrays(*(np.asarray(x, float) for x in (tox, Na, Ndp, Dit)))
    phiF = KT * np.log(Na / NI)
    Cox = EOX / tox
    Wd = np.sqrt(2 * ES * 2 * phiF / (Q * Na))
    Qd, Cd = Q * Na * Wd, ES / Wd
    phims = -(EG2 + phiF) + np.maximum(0.0, KT * np.log(NC / Ndp))     # n+ poly — 축퇴가 아니면 일함수가 밀린다
    Vpoly = Qd ** 2 / (2 * Q * ES * Ndp)                               # 문턱에서 poly 공핍 전압
    Vt = phims + 2 * phiF + Qd / Cox + Q * Dit * phiF / Cox + Vpoly     # 계면 트랩: 중간 띠 ~ 문턱 사이가 채워진 전하
    Ceff = 1 / (1 / Cox + Qd / (Q * Ndp * ES))
    SS = SS0 * (1 + (Cd + Q * Dit) / Ceff)
    Vov = np.maximum(VDD - Vt, 1e-3)
    Cinv = 1 / (1 / Cox + (Cox * Vov + Qd) / (Q * Ndp * ES))           # 켜진 상태 poly 공핍
    return Vt, SS, Cinv * Vov ** ALPHA, Cox


def _solve(f, lo, hi, target, log):
    """단조 함수 f의 역 — 배열 target마다 이분법. f가 감소 함수여도 된다."""
    target = np.asarray(target, float)
    a = np.full(target.shape, np.log(lo) if log else lo)
    b = np.full(target.shape, np.log(hi) if log else hi)
    inc = f(np.exp(b) if log else b) > f(np.exp(a) if log else a)
    for _ in range(80):
        m = (a + b) / 2
        v = f(np.exp(m) if log else m)
        go_up = (v < target) == inc
        a, b = np.where(go_up, m, a), np.where(go_up, b, m)
    m = (a + b) / 2
    return np.exp(m) if log else m


NOM["Na"] = float(_solve(lambda n: device(NOM["tox"], n, NOM["Ndp"], NOM["Dit"])[0], 1e16, 3e18, VT_TARGET, True))
_ION0 = float(device(NOM["tox"], NOM["Na"], NOM["Ndp"], NOM["Dit"])[2])


def wat(tox, Na, Ndp, Dit, sd=1.0, dT=0.0, c49=0.0) -> dict:
    """공정값 → WAT 값 — Vt [V] · SS [mV/dec] · on 전류(기준 = 1) · C-V 산화막 [Å] · 면저항 셋 [Ω/sq]."""
    Vt, SS, Ion, Cox = device(tox, Na, Ndp, Dit)
    sd, dT, c49, Ndp = (np.asarray(x, float) for x in (sd, dT, c49, Ndp))
    return {"Vt": Vt, "SS": SS, "ion": Ion / _ION0, "tox_cv": EOX / Cox * 1e8,
            "rs_sd": RS_SD0 / sd - DRS_DT * dT,
            "rs_poly": RS_POLY0 * (1e20 / Ndp),
            "rs_sil": RS_SIL0 * (1 + c49 * (RHO_RATIO - 1))}


BASE = {k: float(v) for k, v in wat(**NOM).items()}
SPEC = {"Vt": (VT_TARGET - 0.10, VT_TARGET + 0.10), "ion": (0.84, 1.16), "SS": (None, BASE["SS"] + 10.0),
        "rs_sd": (RS_SD0 * 0.85, RS_SD0 * 1.15), "rs_poly": (RS_POLY0 * 0.85, RS_POLY0 * 1.15),
        "rs_sil": (None, RS_SIL0 * 1.15)}


def in_spec(meas: dict) -> dict:
    out = {}
    for k, (lo, hi) in SPEC.items():
        v = np.asarray(meas[k])
        out[k] = (v >= (lo if lo is not None else -np.inf)) & (v <= (hi if hi is not None else np.inf))
    return out


def sample_normal(rng, n: int) -> dict:
    """정상 웨이퍼 n장의 공정값 — 기준 ± 정상 산포."""
    return {"tox": NOM["tox"] * (1 + SPREAD["tox"] * rng.standard_normal(n)),
            "Na": NOM["Na"] * (1 + SPREAD["Na"] * rng.standard_normal(n)),
            "Ndp": NOM["Ndp"] * np.maximum(0.2, 1 + SPREAD["Ndp"] * rng.standard_normal(n)),
            "Dit": NOM["Dit"] * np.exp(SPREAD["Dit"] * rng.standard_normal(n)),
            "sd": NOM["sd"] * (1 + SPREAD["sd"] * rng.standard_normal(n)),
            "dT": NOM["dT"] + SPREAD["dT"] * rng.standard_normal(n),
            "c49": np.abs(SPREAD["c49"] * rng.standard_normal(n))}


def _noisy(k, value, rng, n=None):
    e = NOISE[k] * rng.standard_normal(n)
    return value * (1 + e) if k in REL_NOISE else value + e


def measure(params: dict, rng) -> dict:
    """공정값 → 측정 잡음이 붙은 WAT 값(웨이퍼 하나)."""
    w = {k: float(v) for k, v in wat(**params).items()}
    return {k: float(_noisy(k, w[k], rng)) for k in MEAS}


def _sigma_normal(n: int = 4000, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    w = wat(**sample_normal(rng, n))
    return {k: float(np.sqrt(np.var(w[k]) + (NOISE[k] * (BASE[k] if k in REL_NOISE else 1)) ** 2)) for k in MEAS}


SIGMA = _sigma_normal()     # 정상 웨이퍼의 WAT 산포(공정 산포 + 측정 잡음) — 「다른 측정과 맞나」의 잣대


def _f(param, meas, fixed):
    def f(x):
        p = dict(fixed)
        p[param] = x
        return wat(**p)[meas]
    return f


def normal_range(param: str) -> tuple:
    """불량 경계 — 이 파라미터 하나만 움직였을 때 측정이 전부 합격선 안인 범위. 한쪽이 끝없이 괜찮으면 None."""
    lo, hi, log = _GRID[param]
    xs = np.geomspace(lo, hi, 4001) if log else np.linspace(lo, hi, 4001)
    p = {k: np.full(xs.shape, v) for k, v in NOM.items()}
    p[param] = xs
    ok = np.logical_and.reduce(list(in_spec(wat(**p)).values()))
    i0 = int(np.argmin(np.abs(np.log(xs / NOM[param]) if log else xs - NOM[param])))
    a = i0
    while a > 0 and ok[a - 1]:
        a -= 1
    b = i0
    while b < len(xs) - 1 and ok[b + 1]:
        b += 1
    return (None if a == 0 else float(xs[a])), (None if b == len(xs) - 1 else float(xs[b]))


_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def fmt(param: str, x) -> str:
    """화면용 — 산화막은 Å, 도핑 · S/D 도즈는 기준 대비 %, poly는 cm⁻³, 계면 트랩은 기준의 몇 배, RTA는 °C, C49는 %."""
    if x is None:
        return "—"
    if param == "tox":
        return f"{x * 1e8:.0f} Å"
    if param == "Na":
        return f"{(x / NOM['Na'] - 1) * 100:+.0f}%"
    if param == "sd":
        return f"{(x - 1) * 100:+.0f}%"
    if param == "dT":
        return f"{x:+.0f} °C"
    if param == "c49":
        return f"{x * 100:.0f}%"
    if param == "Ndp":
        e = int(np.floor(np.log10(x)))
        return f"{x / 10 ** e:.1f}×10{str(e).translate(_SUP)} cm⁻³"
    r = x / NOM["Dit"]
    return "기준 이하" if r < 1 else f"×{r:.1f}" if r < 10 else f"×{r:.0f}"


def fmt_range(param: str, lo, hi) -> str:
    if lo is None and hi is None:
        return "제한 없음"
    if lo is None:
        return f"{fmt(param, hi)} 이하"
    if hi is None:
        return f"{fmt(param, lo)} 이상"
    return f"{fmt(param, lo)} ~ {fmt(param, hi)}"


def _tox_fixed(meas, rng, n):
    """다른 파라미터를 의심할 때 산화막을 무엇으로 둘까 — C-V 값이 정상 산포 안이면 그 값(산포를 줄인다), 밖이면 기준값.
    밖인데 기준값으로 두는 것은 「이 파라미터 하나가 혼자 벌인 일」이라는 가정이라서다 — C-V가 안 맞는 것은 아래 대조가 잡는다."""
    if abs(meas["tox_cv"] - BASE["tox_cv"]) <= 4 * SIGMA["tox_cv"]:
        return meas["tox_cv"] * 1e-8 * (1 + NOISE["tox_cv"] * rng.standard_normal(n)), meas["tox_cv"] * 1e-8
    return NOM["tox"] * (1 + SPREAD["tox"] * rng.standard_normal(n)), NOM["tox"]


def estimate(cause_id: str, meas: dict, n: int = 2000, seed: int = 0) -> dict | None:
    """원인 하나 → 그 파라미터가 얼마나 벗어났나(중앙값 · 90% 범위) + 그 크기면 나와야 할 다른 측정이 이번과 맞나.
    크기를 재는 식이 없는 원인이면 None.

    산화막은 C-V 값을 그대로 쓰고(WAT 표준 항목), 채널 도핑은 Vt, 계면 트랩은 SS, poly 도핑 · S/D 도즈 · RTA 온도 · C49는
    면저항(막은 패턴 · silicide 패턴)을 거꾸로 푼다. 90% 범위는 다른 파라미터의 정상 산포와 측정 잡음을 n번 뽑아 같은 역산을
    되풀이한 퍼짐이다."""
    param = CAUSE_KNOB.get(cause_id)
    if param is None:
        return None
    rng = np.random.default_rng(seed)
    if param == "tox":
        xs = meas["tox_cv"] * 1e-8 * (1 + NOISE["tox_cv"] * rng.standard_normal(n))
        tox0 = None
    else:
        nuis = sample_normal(rng, n)
        tox_s, tox0 = _tox_fixed(meas, rng, n)
        mkey = KNOB[param][1]
        target = _noisy(mkey, meas[mkey], rng, n)
        fixed = {**nuis, "tox": tox_s}
        lo, hi, log = _GRID[param]
        xs = _solve(_f(param, mkey, fixed), lo, hi, target, log)
    p5, p50, p95 = (float(v) for v in np.percentile(xs, [5, 50, 95]))
    blo, bhi = normal_range(param)
    out = (blo is not None and p50 < blo) or (bhi is not None and p50 > bhi)
    p = dict(NOM)
    if tox0 is not None:
        p["tox"] = tox0
    p[param] = p50
    pred = {k: float(v) for k, v in wat(**p).items()}
    ok = {k: bool(abs(pred[k] - meas[k]) <= 4 * SIGMA[k]) for k in MEAS}
    return {"param": param, "label": KNOB[param][0], "from": MEAS_LABEL[KNOB[param][1]],
            "value": fmt(param, p50), "lo": fmt(param, p5), "hi": fmt(param, p95),
            "normal": fmt_range(param, blo, bhi), "out_of_range": bool(out),
            "consistent": all(ok.values()), "mismatch": [MEAS_LABEL[k] for k in MEAS if not ok[k]],
            "pred": pred, "raw": {"p5": p5, "p50": p50, "p95": p95}}


def fingerprint(meas: dict, causes=None) -> list:
    """파라미터를 하나씩 의심해 보기 — 이번 WAT 값으로 계산한 크기, 정상 범위 밖인가, 그 크기면 나와야 할 다른 측정이 이번과
    맞나. 「밖이고 맞는」 줄이 이번 사고의 설명이다. causes를 주면(엔진이 맵 모양으로 남긴 후보) 그 원인들의 파라미터만 본다 —
    막은 S/D 면저항 하나로는 도즈 부족과 RTA 온도 부족이 같아 보여서, 둘은 맵 모양이 가른다."""
    rep = {}
    for c, prm in CAUSE_KNOB.items():
        if causes is None or c in causes:
            rep.setdefault(prm, []).append(c)
    rows = []
    for prm, cs in rep.items():
        e = estimate(cs[0], meas)
        rows.append({**e, "causes": cs, "explains": e["out_of_range"] and e["consistent"]})
    return rows
