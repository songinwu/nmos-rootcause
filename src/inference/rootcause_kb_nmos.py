"""
원인 공정 추천 — NMOS 지식표 (2026-10-28 초안, wiki 「NMOS 원인 공정 지식표 — 조원 설명용」 2판의 코드판).

목표는 「38단계 중 하나가 잘못됐을 때 웨이퍼맵 모양 + 칩별 측정으로 바로 짚기」다. 엔진은 문헌 CMOS KB와 같고
(rootcause_kb.score_batch에 이 모듈의 prior · q를 넘긴다) 검사 칸과 원인 표만 다르다.

    무엇을 만드나 : P형 (100) 기판 위 NMOS 하나 — STI · LDD 없음 · salicide · W contact · Al metal 1 (마스크 5장)
    공정 흐름     : IDEC 0.5 µm CMOS 108스텝에서 NMOS 부분만 + STI(KSIA §2.7) — 38단계 (STEPS)
    검사 칸 11개  : 칩마다 재는 측정 9가지(WAT 방식)에서 Vt를 높음 · 낮음으로 나눈 것. 칩은 가장 크게 벗어난
                    측정 하나로 판정한다. BIN_GROUP4로 기존 4칸(끊김 · 누설 · 문턱 이탈 · 전류 부족)에 묶인다.

⚠️ 한계
- 원인 40개 중 문헌 원문 사례가 직접 있는 것은 18개(evidence 「원문」 16 · 「일부」 2), 다른 공정 사례를 옮긴 것 4개
  (「옮김」), 나머지 18개는 강의 · 소자 원리(「강의 · 원리」)다.
- 측정 방향은 원인마다 「바뀌는 측정」만 적었고 세기(%)는 모른다 — 엔진 CMOS KB와 같은 logistic-normal 기댓값.
- 맵 모양은 WM-811K(로직 웨이퍼) 실제 맵을 빌려 쓴다.

torch를 import하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.inference import rootcause_kb as kb

PATTERNS = kb.PATTERNS
BIN_TYPES = ["open", "gate_leak", "junction_leak", "iso_leak", "metal_leak",
             "vt_high", "vt_low", "ss", "ion_low", "rs_sd", "rs_poly"]
BIN_LABELS = {
    "open": "연결 끊김", "gate_leak": "gate 누설", "junction_leak": "접합 · off 누설", "iso_leak": "분리 누설",
    "metal_leak": "배선 누설", "vt_high": "Vt 높음", "vt_low": "Vt 낮음", "ss": "SS 기울기 큼",
    "ion_low": "on 전류 부족", "rs_sd": "S/D 면저항 큼", "rs_poly": "poly 면저항 큼",
}
BIN_GROUP4 = {"open": "끊김", "gate_leak": "누설", "junction_leak": "누설", "iso_leak": "누설", "metal_leak": "누설",
              "vt_high": "문턱 이탈", "vt_low": "문턱 이탈", "ss": "문턱 이탈",
              "ion_low": "전류 부족", "rs_sd": "전류 부족", "rs_poly": "전류 부족"}
GROUP4 = ["끊김", "누설", "문턱 이탈", "전류 부족"]
B = len(BIN_TYPES)

STEPS = {f"{i:02d}": n for i, n in enumerate([
    "입고", "세정", "pad oxide", "질화막 증착", "활성 영역 포토", "질화막 · pad 식각", "trench 식각", "PR 제거",
    "liner 산화", "trench 채우기", "치밀화 열처리", "STI CMP", "질화막 제거", "pad oxide 제거", "screen oxide",
    "Vt 주입", "screen oxide 제거", "세정", "gate 산화", "poly 증착", "gate 포토", "poly 식각", "PR 제거",
    "S/D 주입", "spacer", "활성화 RTA", "salicide", "ILD 증착", "ILD CMP", "contact 포토", "contact 식각",
    "PR 제거", "TiN + W", "W CMP", "metal 1", "보호막 · pad", "forming gas", "EDS"], start=1)}


@dataclass(frozen=True)
class NCause:
    id: str
    name: str
    steps: tuple        # STEPS 키 — 공통 원인은 ()
    knob: str           # 무엇이 틀어지나
    primary: tuple      # 바뀌는 측정 (BIN_TYPES)
    pattern: str        # 맵 모양 (WM-811K 이름)
    evidence: str       # 원문 · 일부 · 옮김 · 강의 · 원리
    source: str


C = NCause
CAUSES = [
    C("n01_bevel", "기판 테두리 + 질화막 박리", ("01", "04"), "긴 bevel · 매끈한 네 귀퉁이 · 질화막이 못 덮음", ("metal_leak",), "edge-loc", "원문", "Ma 2024 (N08)"),
    C("n02_clean_metal", "세정 금속 오염", ("02",), "금속 오염이 남아 고온에서 확산 → 트랩", ("junction_leak", "gate_leak"), "near-full", "강의", "IDEC 강의"),
    C("n05_active_focus", "활성 영역 포토 가장자리 초점", ("05",), "가장자리 초점 → 활성 폭 변함", ("ion_low",), "edge-ring", "원리", "TI 특허 초점 문제를 옮김"),
    C("n06_nit_etch_under", "질화막 · pad 식각 부족", ("06",), "잔막 → trench 얕음", ("iso_leak",), "near-full", "강의", "IDEC 30% 과식각"),
    C("n07_trench_edge", "trench 식각 가장자리 얕음", ("07",), "edge ring → 가장자리 분리 부족", ("iso_leak",), "edge-ring", "강의", "KSIA STI · Chai · Brady 옮김"),
    C("n08_pr_residue", "STI 포토 뒤 PR 찌꺼기", ("08",), "찌꺼기 → 다음 막 결함", ("metal_leak", "open"), "loc", "원리", "—"),
    C("n09_liner_skip", "liner 산화 생략", ("09",), "trench 모서리가 날카로움", ("junction_leak",), "near-full", "원리", "일반 지식"),
    C("n10_fill_void", "trench 채우기 빈 곳 · 입자", ("10",), "void · 챔버 입자 → 분리 결함", ("iso_leak",), "loc", "원문", "Hansen · Stapper (N09)"),
    C("n11_densify_skip", "치밀화 열처리 생략", ("11",), "막이 물러 HF에 깎임 → 모서리 파임", ("junction_leak",), "near-full", "원리", "일반 지식"),
    C("n12_sti_cmp_edge", "STI CMP 가장자리 과연마", ("12",), "STI가 낮아져 모서리 누설", ("junction_leak",), "edge-ring", "옮김", "Goh · Chen 옮김"),
    C("n13_nit_residue", "질화막 잔류", ("13",), "잔류 질화막 → 그 자리 산화 이상", ("vt_high", "ion_low"), "loc", "원리", "—"),
    C("n14_padox_overetch", "pad oxide 과식각 (divot)", ("14",), "STI 모서리 파임", ("junction_leak",), "near-full", "원리", "일반 지식"),
    C("n15_screen_skip", "screen oxide 생략 (채널링)", ("15",), "채널링 → Vt가 칩마다 흩어짐", ("vt_high", "vt_low"), "near-full", "강의", "IDEC 채널링 방지"),
    C("n16_vt_dose", "Vt 주입 도즈 이탈", ("16",), "채널 도핑이 틀어짐", ("vt_high",), "near-full", "원문", "Stapper 1983 (N13)"),
    C("n17_screen_residue", "screen oxide 잔막", ("17",), "gate 산화막이 두꺼워짐", ("vt_high", "ion_low"), "near-full", "원리", "—"),
    C("n18_rinse_charge", "gate 앞 세정 헹굼 정전기", ("18",), "물 헹굼 마찰 전기 → 산화막 약화", ("gate_leak",), "center", "원문", "Lin 2012 (N05)"),
    C("n19_gox_thick", "gate 산화 두께 이탈", ("19",), "시간 · 온도 이탈 → 두꺼워짐", ("vt_high", "ion_low"), "near-full", "원문", "Stapper 1983 (N14)"),
    C("n20_poly_doping", "poly 도핑 부족", ("20",), "poly 공핍", ("rs_poly", "vt_high"), "near-full", "원리", "—"),
    C("n21a_gate_focus", "gate 포토 가장자리 초점", ("21",), "gate 길이 변함", ("vt_high", "ion_low"), "edge-ring", "일부", "TI 특허 · Goh (N02)"),
    C("n21b_develop_residue", "gate 포토 현상 찌꺼기", ("21",), "헹굼물 오염 → PR 재부착", ("metal_leak", "open"), "donut", "원문", "Ng 2009 (N17)"),
    C("n22_poly_overetch", "poly 식각 가장자리 과식각", ("22",), "gate 산화막 손상 (edge ring)", ("gate_leak",), "edge-ring", "강의", "KSIA 선택비 · Chai · Brady 옮김"),
    C("n23_gate_ash_charge", "gate 뒤 ashing 대전", ("23",), "poly 안테나로 대전 손상", ("gate_leak",), "center", "옮김", "Chen 2000"),
    C("n24_sd_dose", "S/D 주입 도즈 이탈", ("24",), "도즈 · 에너지 이탈", ("rs_sd", "ion_low"), "near-full", "원문", "Stapper 1983"),
    C("n25_spacer_thin", "spacer 얇음 (salicide 다리)", ("25",), "gate와 S/D가 TiSi2로 이어짐", ("gate_leak", "open"), "near-full", "원리", "일반 지식"),
    C("n26_rta_edge", "활성화 RTA 가장자리 온도", ("26",), "가장자리 활성화 부족", ("rs_sd", "ion_low"), "edge-ring", "원문", "Hansen · Micron (N01)"),
    C("n27_salicide_temp", "salicide 열처리 온도 이탈", ("27",), "고저항 상", ("rs_sd", "rs_poly"), "near-full", "강의", "KSIA salicide"),
    C("n28_ild_particle", "ILD 챔버 입자", ("28",), "입자 낙하", ("metal_leak", "open"), "loc", "원문", "Hansen · Stapper (N09)"),
    C("n29_ild_cmp_edge", "ILD CMP 가장자리 두께", ("29",), "가장자리 ILD 두꺼움 → contact 안 열림", ("open",), "edge-ring", "원문", "Goh 2001 (N04)"),
    C("n30_contact_barc", "contact 반사방지막 두꺼움", ("30",), "BT 부족 → 3 · 7시 안 열림", ("open",), "edge-loc", "원문", "Liu 2019 (N07)"),
    C("n31_contact_edge_ring", "contact 식각 edge ring", ("31",), "새 ring → 가장자리 덜 뚫림", ("open",), "edge-ring", "원문", "Chai · Brady (N03)"),
    C("n32_contact_ash_charge", "contact 뒤 ashing 대전", ("32",), "금속 안테나로 대전 손상", ("gate_leak",), "center", "원문", "Chen 2000 (N06)"),
    C("n33_tin_thin", "TiN 장벽 얇음", ("33",), "W 증착 때 접합 손상", ("junction_leak",), "near-full", "원리", "—"),
    C("n34_w_cmp_edge", "W CMP 가장자리 덜 연마", ("34",), "W 잔류 → 배선 붙음", ("metal_leak",), "edge-ring", "옮김", "Goh 옮김"),
    C("n35_metal_underetch", "metal 1 식각 부족", ("35",), "배선 붙음", ("metal_leak",), "near-full", "원리", "—"),
    C("n36_pad_open", "pad 덜 열림", ("36",), "측정 안 됨", ("open",), "near-full", "원리", "—"),
    C("n37_forming_gas_skip", "forming gas 생략", ("37",), "계면 결함", ("ss", "vt_high"), "near-full", "옮김", "IDEC · Mack 2023 (N15)"),
    C("n38_probe", "검사 바늘 접촉", ("38",), "바늘 접촉 불량", ("open",), "near-full", "일부", "IBM · TI 특허 (N16)"),
    C("c_cmp_scratch", "CMP 긁힘", ("12", "29", "34"), "뭉친 연마 입자", ("metal_leak", "open"), "scratch", "원문", "Kwon 2013 (N11)"),
    C("c_robot_scratch", "이송 로봇 긁힘", (), "기운 팔", ("metal_leak", "open"), "scratch", "원문", "TSMC 특허 (N12)"),
    C("c_cleanroom", "클린룸 입자", (), "입자 밀도 증가", ("metal_leak", "open"), "random", "원문", "Hansen · Stapper (N10)"),
]
UNOBSERVABLE = {"03": "pad oxide 두께 — 칩 측정에 거의 안 나타남, 공정 중 두께 측정으로만"}

K = len(CAUSES)
CAUSE_IDS = [c.id for c in CAUSES]
CAUSE_INDEX = {c.id: i for i, c in enumerate(CAUSES)}
PATTERN_CANDIDATES = {p: [c.id for c in CAUSES if c.pattern == p] for p in PATTERNS[1:]}

assert len(set(CAUSE_IDS)) == K
assert all(b in BIN_TYPES for c in CAUSES for b in c.primary)
assert all(c.pattern in PATTERNS[1:] for c in CAUSES)
assert all(s in STEPS for c in CAUSES for s in c.steps)
assert all(PATTERN_CANDIDATES[p] for p in PATTERNS[1:]), "후보가 없는 패턴이 있다"

# 칸이 11개라 CMOS KB(4칸, 주 1.5)보다 주 logit을 키워 주 칸 몫을 비슷하게 맞춘다 — 주 하나면 약 50%
LOGIT_PRIMARY = 2.5
SIGMA = kb.DIRECTION_SIGMA
CHI2_90 = 15.987        # χ²(df = 10) 90백분위 — 「무작위와 구별되나」 기준 (CMOS KB는 df = 3의 6.251)


def direction_logits(c: NCause) -> np.ndarray:
    z = np.zeros(B)
    for b in c.primary:
        z[BIN_TYPES.index(b)] = LOGIT_PRIMARY
    return z


def expected_signature(logits, sigma: float = SIGMA, n: int = 20000, seed: int = 0) -> np.ndarray:
    z = np.asarray(logits, float) + np.random.default_rng(seed).normal(0.0, sigma, (n, len(logits)))
    w = np.exp(z - z.max(axis=1, keepdims=True))
    return (w / w.sum(axis=1, keepdims=True)).mean(axis=0)


SIGNATURES = np.array([expected_signature(direction_logits(c)) for c in CAUSES])
assert np.allclose(SIGNATURES.sum(axis=1), 1.0)

IN_TABLE = np.zeros((len(PATTERNS), K), dtype=bool)
for _p, _row in PATTERN_CANDIDATES.items():
    IN_TABLE[PATTERNS.index(_p), [CAUSE_INDEX[_c] for _c in _row]] = True


def prior_matrix(smooth: float = kb.PRIOR_SMOOTH) -> np.ndarray:
    """(9, K) — P(원인 | 패턴), 후보 균등 + 평활. none 행은 0."""
    M = np.zeros((len(PATTERNS), K))
    for p, row in PATTERN_CANDIDATES.items():
        i = PATTERNS.index(p)
        M[i, [CAUSE_INDEX[c] for c in row]] = 1.0
        M[i] /= M[i].sum()
        M[i] = (1.0 - smooth) * M[i] + smooth / K
    return M


def bin_likelihood(signatures=None, robust: float = kb.BIN_ROBUST) -> np.ndarray:
    S = SIGNATURES if signatures is None else np.asarray(signatures, float)
    return (1.0 - robust) * S + robust / S.shape[1]


def informative(counts) -> np.ndarray:
    """(N,) bool — 측정 결과가 무작위(균등)와 구별되나. 아니면 측정 항을 버린다."""
    X = np.atleast_2d(np.asarray(counts, float))
    E = X.sum(axis=1, keepdims=True) / X.shape[1]
    chi = (((X - E) ** 2) / np.maximum(E, 1e-12)).sum(axis=1)
    thr = CHI2_90 if X.shape[1] == B else kb.BIN_UNIFORM_CHI2
    return (X.sum(axis=1) > 0) & (chi >= thr)


def score(pattern_probs, counts, signatures=None, guard: bool = True) -> np.ndarray:
    """(N, K) 원인 로그 점수 — 엔진(rootcause_kb.score_batch)에 이 표의 prior · q를 넘긴다."""
    q = bin_likelihood(signatures)
    X = np.atleast_2d(np.asarray(counts, float))
    if guard:
        X = np.where(informative(X)[:, None], X, 0.0)
    return kb.score_batch(pattern_probs, X, prior=prior_matrix(), q=q)


def eligible(post: np.ndarray, top_pattern) -> np.ndarray:
    post = np.atleast_2d(post)
    return IN_TABLE[np.asarray(top_pattern).reshape(-1)] | (post * 100 >= kb.MIN_OFFTABLE_PCT)


def collapse4(counts) -> np.ndarray:
    """(N, 11) → (N, 4) — 기존 4칸으로 묶기."""
    X = np.atleast_2d(np.asarray(counts, float))
    out = np.zeros((X.shape[0], 4))
    for j, b in enumerate(BIN_TYPES):
        out[:, GROUP4.index(BIN_GROUP4[b])] += X[:, j]
    return out


SIGNATURES4 = collapse4(SIGNATURES)


def predicted_unique(group4: bool = False) -> dict:
    """「모양 + 바뀌는 측정」 조합이 다른 원인과 겹치지 않는가 — 표만 보고 하는 예측."""
    keys = {}
    for c in CAUSES:
        m = tuple(sorted({BIN_GROUP4[b] for b in c.primary} if group4 else c.primary))
        keys.setdefault((c.pattern, m), []).append(c.id)
    return {cid: len(v) == 1 for v in keys.values() for cid in v}
