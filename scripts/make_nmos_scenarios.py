"""
NMOS 시연 시나리오 → data/demo/nmos_scenarios.json (대시보드 섹션 16 「NMOS 41단계」 — 38단계 + SAB 3단계)

맵은 실제 WM-811K(A-1b lot 단위 분할의 test lot — 패턴 CNN이 한 번도 안 본 맵)이고, 측정 결과(11칸)는 NMOS 지식표
(src/inference/rootcause_kb_nmos.py)의 원인에서 합성한다 — N-1 합성 시험과 같은 방식(주 측정 비율 = 지식표 기댓값,
칩의 10%는 아무 칸 — 잡음). CNN은 맵을 실제로 판정하고, 엔진은 NMOS 지식표로 순위를 매긴다. 같은 측정을 기존 4칸으로
묶었을 때의 공동 1순위 수도 같이 담는다(「측정을 늘려야 짚인다」를 화면에서 보이려고).

*(2026-10-28 보강 — 「얼마나」)* 웨이퍼마다 WAT 값(Vt · SS · on 전류 · C-V 산화막)을 NMOS 소자 식(src/inference/nmos_device.py)
으로 만든다 — 기준값 ± 정상 산포에, 시나리오가 정한 사고 크기(PLANT)를 얹는다. 엔진이 고른 원인마다 그 파라미터가 얼마나
벗어났는지(중앙값 · 90% 범위)와 「파라미터를 하나씩 의심해 보기」 표를 같이 담는다. 원인마다 「크기를 말할 수 있는 유형인가」
(KIND)도 카드에 붙인다.
결정론적 — 같은 캐시·체크포인트면 같은 파일.

⚠️ 심을 때와 채점할 때 같은 표를 쓴다 — 「표가 맞다면 엔진이 이렇게 짚는다」까지다(N-1과 같은 한계).

    conda run -n semi_ai python -m scripts.make_nmos_scenarios
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.a1_lot_consistency import load_cache                     # noqa: E402
from scripts.a1b_lotwise_model import split_by_lot                     # noqa: E402
from scripts.make_rootcause_scenarios import cnn_probs                 # noqa: E402
from src.inference import nmos_device as nd                           # noqa: E402
from src.inference import rootcause_kb as kb                           # noqa: E402
from src.inference import rootcause_kb_nmos as nk                      # noqa: E402

OUT = "data/demo/nmos_scenarios.json"
SEED = 2028
EPS_BIN = 0.10

PAT_KO = {"center": "가운데", "donut": "도넛", "edge-loc": "가장자리 일부", "edge-ring": "가장자리 링",
          "loc": "한 곳에 뭉침", "near-full": "거의 전체", "random": "흩어짐", "scratch": "선"}

# 만드는 순서를 화면에서 묶어 보이는 단위 — 38줄을 다 늘어놓으면 스크롤해야 한다
GROUPS = [("준비", "01", "02"), ("STI 소자 분리", "03", "14"), ("채널 · gate", "15", "23"),
          ("source · drain · SAB", "24", "26c"), ("salicide", "27", "27"), ("contact · 배선", "28", "36"),
          ("마무리 · 검사", "37", "38")]

# 측정 9가지(12칸 — 면저항은 패턴 셋) — 칩은 가장 크게 벗어난 측정 하나로 판정한다
BIN_PLAIN_TEST = {
    "open": "contact 체인 저항이 끝없이 큼 — 어딘가 안 이어짐",
    "gate_leak": "gate에 전압을 걸면 gate로 전류가 샘",
    "junction_leak": "꺼 둔 상태(Vg = 0)에서 drain 전류가 흐름",
    "iso_leak": "STI 너머 이웃 활성 영역으로 전류가 흐름",
    "metal_leak": "이웃 배선 사이로 전류가 흐름",
    "vt_high": "문턱 전압이 기준보다 높음",
    "vt_low": "문턱 전압이 기준보다 낮음",
    "ss": "켜질 때 전류가 늘어나는 기울기가 완만함",
    "ion_low": "켜 둔 상태 전류가 기준보다 작음",
    "rs_sd": "silicide를 막은 시험 패턴(SAB)의 S/D 면저항이 기준보다 큼",
    "rs_poly": "silicide를 막은 poly 저항의 면저항이 기준보다 큼",
    "rs_sil": "silicide가 덮인 패턴의 면저항이 기준보다 큼",
}
# 색은 기존 4칸 묶음별 계열 — 끊김 보라 · 누설 붉은 계열 · 문턱 노랑~초록 · 전류 파랑 계열
COLORS = ["#8F7EF2", "#E85D5D", "#F2946B", "#D9579A", "#B23A48",
          "#E0A800", "#B5C930", "#6FBF5A", "#2E9FD0", "#3FC1C0", "#6C86E8", "#9BB7F5"]

# 엔진이 말하는 것(suspect) · 왜 이 모양(shape, None이면 모양별 기본 문장) · 먼저 확인할 것(checks)
TEXT = {
    "n01_bevel": ("입고 기판 테두리가 길어 질화막이 못 덮고 벗겨졌을 가능성",
                  "벗겨진 조각이 테두리 한쪽에서 떨어져 그 근처 칩만 걸려요 (Ma 2024)",
                  ["입고 기판 테두리 모양", "질화막 테두리 덮임 (현미경)"]),
    "n02_clean_metal": ("세정에서 금속 오염이 남아 열처리 때 안으로 퍼졌을 가능성", None,
                        ["세정액 교체 주기 기록", "표면 금속 오염 분석"]),
    "n05_active_focus": ("활성 영역 포토 가장자리 초점이 어긋나 활성 폭이 변했을 가능성",
                         "가장자리는 웨이퍼가 휘어 초점이 잘 어긋나 링으로 나와요",
                         ["가장자리 활성 폭 (CD)", "노광기 초점 기록"]),
    "n06_nit_etch_under": ("질화막 · pad 식각이 덜 돼 trench가 얕아졌을 가능성", None,
                           ["trench 깊이", "식각 종점 기록"]),
    "n07_trench_edge": ("trench 식각이 가장자리에서 얕아 분리가 부족했을 가능성",
                        "식각 장비 edge ring이 닳으면 가장자리만 달라져 링으로 나와요 (Chai · Brady 옮김)",
                        ["가장자리 trench 깊이", "edge ring 사용 시간"]),
    "n08_pr_residue": ("STI 포토 뒤 PR 찌꺼기가 남아 다음 막에 결함을 만들었을 가능성",
                       "찌꺼기가 남은 자리만 걸려 한 곳에 뭉쳐요",
                       ["공정 중 결함 검사 맵을 단계별로 겹쳐 보기", "ashing 기록"]),
    "n09_liner_skip": ("liner 산화가 빠지거나 얇아 trench 모서리가 날카로웠을 가능성", None,
                       ["STI 모서리 단면", "liner 두께 기록"]),
    "n10_fill_void": ("trench 채우기에서 빈 곳(void)이나 챔버 입자가 생겼을 가능성",
                      "입자가 떨어진 자리만 걸려 한 곳에 뭉쳐요 (Hansen · Stapper)",
                      ["채우기 뒤 결함 검사 맵", "챔버 세정 주기"]),
    "n11_densify_skip": ("치밀화 열처리가 빠져 채운 막이 HF에 많이 깎였을 가능성", None,
                         ["채운 막 식각 속도", "STI 모서리 단면"]),
    "n12_sti_cmp_edge": ("STI CMP가 가장자리를 너무 깎아 모서리 누설이 생겼을 가능성",
                         "CMP는 가장자리 압력 · 속도가 달라 링으로 나와요 (Goh · Chen 옮김)",
                         ["가장자리 STI 높이", "CMP 가장자리 압력 기록"]),
    "n13_nit_residue": ("질화막이 덜 벗겨져 그 자리 gate 산화가 이상했을 가능성",
                        "덜 벗겨진 자리만 걸려 한 곳에 뭉쳐요", ["질화막 제거 뒤 표면 검사", "인산 식각 시간"]),
    "n14_padox_overetch": ("pad oxide를 너무 벗겨 STI 모서리가 파였을(divot) 가능성", None,
                           ["STI 모서리 단면", "HF 식각 시간 기록"]),
    "n15_screen_skip": ("screen oxide가 빠지거나 얇아 Vt 주입이 채널링됐을 가능성", None,
                        ["screen oxide 두께 기록", "칩별 Vt 흩어짐"]),
    "n16_vt_dose": ("Vt 주입 도즈가 어긋났을 가능성", None, ["이온 주입기 도즈 기록", "모니터 웨이퍼 면저항"]),
    "n17_screen_residue": ("screen oxide가 덜 벗겨져 gate 산화막이 두꺼워졌을 가능성", None,
                           ["gate 산화막 두께 (공정 중 측정)", "screen oxide 제거 시간 기록"]),
    "n18_rinse_charge": ("gate 앞 세정의 물 헹굼 정전기로 gate 산화막이 약해졌을 가능성",
                         "회전 헹굼에서 가운데에 전하가 몰려 가운데 칩이 걸려요 (Lin 2012)",
                         ["헹굼물 저항 · CO₂ 주입 기록"]),
    "n19_gox_thick": ("gate 산화 시간 · 온도가 어긋나 두께가 변했을 가능성", None,
                      ["gate 산화막 두께 (공정 중 측정)", "산화로 온도 기록"]),
    "n20_poly_doping": ("poly 도핑이 부족해 gate가 공핍됐을 가능성", None, ["poly 면저항 모니터", "poly 도핑 기록"]),
    "n21a_gate_focus": ("gate 포토 가장자리 초점이 어긋나 gate 길이가 변했을 가능성",
                        "가장자리는 웨이퍼가 휘어 초점이 잘 어긋나 링으로 나와요 (TI 특허 · Goh)",
                        ["가장자리 gate 길이 (CD)", "노광기 초점 기록"]),
    "n21b_develop_residue": ("gate 포토 현상 헹굼물이 오염돼 PR 찌꺼기가 다시 붙었을 가능성",
                             "회전 현상에서 헹굼물이 특정 반경에 남아 고리(도넛)로 나와요 (Ng 2009)",
                             ["현상 헹굼물 상태", "현상 뒤 결함 검사"]),
    "n22_poly_overetch": ("poly 식각이 가장자리에서 과해 gate 산화막이 손상됐을 가능성",
                          "식각 장비 edge ring이 가장자리 plasma를 바꿔 링으로 나와요 (Chai · Brady 옮김)",
                          ["가장자리 gate 산화막 잔막", "edge ring 사용 시간"]),
    "n23_gate_ash_charge": ("gate 뒤 ashing에서 poly가 안테나가 되어 대전 손상을 입었을 가능성",
                            "plasma 전하가 가운데에 몰려 가운데 칩이 걸려요 (Chen 2000 옮김)",
                            ["gate 뒤 ashing 장비 균일도", "안테나 크기별 누설"]),
    "n24_sd_dose": ("S/D 주입 도즈 · 에너지가 어긋났을 가능성", None, ["이온 주입기 도즈 기록", "S/D 면저항 모니터"]),
    "n25_spacer_thin": ("spacer가 너무 얇아 salicide가 gate와 S/D를 이었을 가능성", None,
                        ["spacer 폭 (단면)", "gate – S/D 사이 누설"]),
    "n26_rta_edge": ("활성화 RTA가 가장자리에서 온도가 낮았을 가능성",
                     "램프 가열은 가장자리가 식기 쉬워 링으로 나와요 (Hansen 1997 · Micron 특허)",
                     ["RTA 가장자리 온도 기록", "가장자리 면저항 맵"]),
    "n27_salicide_temp": ("salicide 열처리 온도가 어긋나 고저항 상이 남았을 가능성", None,
                          ["salicide RTA 온도 기록", "면저항 모니터"]),
    "n28_ild_particle": ("ILD 증착 챔버에서 입자가 떨어졌을 가능성",
                         "입자가 떨어진 자리만 걸려 한 곳에 뭉쳐요 (Hansen · Stapper)",
                         ["공정 중 결함 검사 맵을 단계별로 겹쳐 보기", "챔버 세정 주기"]),
    "n29_ild_cmp_edge": ("ILD CMP 뒤 가장자리 ILD가 두꺼워 contact가 안 열렸을 가능성",
                         "CMP 가장자리 연마 속도 차이로 가장자리 두께가 달라 링으로 나와요 (Goh 2001)",
                         ["가장자리 ILD 두께"]),
    "n30_contact_barc": ("contact 포토 반사방지막이 두꺼워 그쪽 contact가 안 열렸을 가능성",
                         "반사방지막이 가장자리 일부(3 · 7시 방향)에서 두꺼워 그쪽만 걸려요 (Liu 2019)",
                         ["반사방지막 두께 맵", "가장자리 contact 저항"]),
    "n31_contact_edge_ring": ("contact 식각 장비 edge ring이 새것이라 가장자리가 덜 뚫렸을 가능성",
                              "edge ring 높이가 가장자리 plasma를 바꿔 링으로 나와요 (Chai · Brady)",
                              ["edge ring 사용 시간 (RF 시간)"]),
    "n32_contact_ash_charge": ("contact 뒤 ashing에서 금속이 안테나가 되어 대전 손상을 입었을 가능성",
                               "plasma 전하가 가운데에 몰려 가운데 칩이 걸려요 (Chen 2000)",
                               ["contact 뒤 ashing 장비 균일도", "안테나 크기별 누설"]),
    "n33_tin_thin": ("TiN 장벽이 얇아 W 증착 때 접합이 손상됐을 가능성", None, ["TiN 두께 기록", "STI 모서리 단면"]),
    "n34_w_cmp_edge": ("W CMP가 가장자리를 덜 깎아 W가 남았을 가능성",
                       "CMP는 가장자리 압력 · 속도가 달라 링으로 나와요 (Goh 옮김)",
                       ["가장자리 W 잔류 검사", "CMP 가장자리 압력 기록"]),
    "n35_metal_underetch": ("metal 1 식각이 덜 돼 배선이 붙었을 가능성", None, ["metal 식각 종점 기록"]),
    "n36_pad_open": ("보호막 pad가 덜 열려 측정이 안 됐을 가능성", None, ["pad 현미경 확인", "다시 재면 살아나나"]),
    "n37_forming_gas_skip": ("forming gas 열처리가 빠져 계면 결함이 남았을 가능성", None,
                             ["forming gas 로 기록"]),
    "n38_probe": ("검사 바늘 접촉이 나빴을 가능성", None, ["다시 재면 살아나나", "바늘 청소 · 교체 기록"]),
    "c_cmp_scratch": ("CMP에서 뭉친 연마 입자가 긁었을 가능성",
                      "패드가 돌며 긁어 곡선(호) 모양이 돼요 (Kwon 2013)",
                      ["긁힘이 곡선인가 (CMP) 직선인가 (로봇)", "연마액 필터 기록"]),
    "c_robot_scratch": ("이송 로봇 팔이 기울어 긁었을 가능성", "팔이 곧게 움직여 직선이 돼요 (TSMC 특허)",
                        ["긁힘이 직선인가", "로봇 팔 수평 점검"]),
    "c_cleanroom": ("클린룸 입자가 늘었을 가능성", "입자는 아무 데나 떨어져 흩어져요 (Hansen · Stapper)",
                    ["클린룸 입자 수 기록"]),
}
# 크기를 어디까지 말하나 — 유형 넷 (wiki §29). 전기 값 중 식을 붙인 것은 nd.CAUSE_KNOB에 있다
KIND = {**{c: "전기 값" for c in ("n02_clean_metal", "n15_screen_skip", "n16_vt_dose", "n17_screen_residue", "n19_gox_thick",
                                   "n20_poly_doping", "n21a_gate_focus", "n24_sd_dose", "n26_rta_edge",
                                   "n27_salicide_temp", "n37_forming_gas_skip")},
        **{c: "여유" for c in ("n05_active_focus", "n06_nit_etch_under", "n07_trench_edge", "n09_liner_skip",
                                "n11_densify_skip", "n12_sti_cmp_edge", "n13_nit_residue", "n14_padox_overetch",
                                "n22_poly_overetch", "n25_spacer_thin", "n29_ild_cmp_edge", "n30_contact_barc",
                                "n33_tin_thin", "n34_w_cmp_edge", "n35_metal_underetch", "n36_pad_open")},
        **{c: "결함 밀도" for c in ("n08_pr_residue", "n10_fill_void", "n28_ild_particle", "c_cmp_scratch",
                                    "c_robot_scratch", "c_cleanroom")},
        **{c: "장비 변수" for c in ("n01_bevel", "n18_rinse_charge", "n21b_develop_residue", "n23_gate_ash_charge",
                                    "n31_contact_edge_ring", "n32_contact_ash_charge", "n38_probe")}}
KIND_TODO = {   # 전기 값인데 크기 식을 아직 안 붙인 것 — 왜
    "n02_clean_metal": "접합 누설 식(트랩 → 누설 전류)을 아직 안 붙였어요",
    "n15_screen_skip": "평균이 아니라 칩마다 흩어지는 사고라, 흩어짐(σ) 식이 따로 필요해요",
    "n21a_gate_focus": "gate 길이 → on 전류 식을 아직 안 붙였어요 (0.5 µm면 Vt는 거의 안 움직여요)",
}


def kind_note(c, checks):
    k = KIND[c.id]
    if k == "전기 값":
        return ("WAT 값으로 크기를 거꾸로 계산해요" if c.id in nd.CAUSE_KNOB else
                f"크기 — {KIND_TODO[c.id]}")
    if k == "여유":
        return f"크기 — 끊김 · 누설은 예 / 아니오라 칩 측정에 크기가 안 남아요. 여유를 넘었는지는 「{checks[0]}」로"
    if k == "결함 밀도":
        return "크기 — 칩당 결함 수로 말할 수 있어요(불량 칩 비율 → 수율 식). 다음 단계에서 붙여요"
    return f"크기 — 칩 측정에 장비 값이 안 남아요. 「{checks[0]}」로 확인"


SHAPE_DEFAULT = {"near-full": "이 단계는 웨이퍼 전체를 한꺼번에 처리해서, 틀어지면 거의 전체 칩이 걸려요",
                 "edge-ring": "가장자리만 조건이 달라져 링으로 나와요",
                 "center": "가운데만 조건이 달라져 가운데 칩이 걸려요",
                 "loc": "한 자리에서만 일어나 한 곳에 뭉쳐요"}
# 측정으로 안 갈리는 묶음 — 무엇을 재면 갈리나 (PDF 2판 4절)
SPLIT = {
    frozenset({"n29_ild_cmp_edge", "n31_contact_edge_ring"}): "가장자리 ILD 두께 vs edge ring 사용 시간(RF 시간)",
    frozenset({"n18_rinse_charge", "n23_gate_ash_charge", "n32_contact_ash_charge"}):
        "헹굼물 저항 · CO₂ 주입 기록 vs ashing 장비 균일도 · 안테나 크기별 누설",
    frozenset({"n17_screen_residue", "n19_gox_thick"}): "screen oxide 제거 시간 기록 vs 산화로 온도 · 시간 기록",
    frozenset({"n36_pad_open", "n38_probe"}): "다시 재 보기(바늘이면 살아남) vs pad 현미경 확인",
}

# (id, 넣을 원인, 제목, 설명)
SCENARIOS = [
    ("N1", "n16_vt_dose", "거의 전체 + Vt 높음",
     "웨이퍼 거의 전체 칩의 Vt가 높게 나왔어요. 「거의 전체」를 만드는 단계는 많지만, Vt만 높은 것은 Vt 주입 하나예요."),
    ("N2", "n37_forming_gas_skip", "거의 전체 + SS 큼 · Vt 높음",
     "N1과 같은 거의 전체인데 SS 기울기도 같이 커졌어요. 측정 하나 차이로 전혀 다른 단계를 짚어요."),
    ("N3", "n20_poly_doping", "거의 전체 + poly 면저항 · Vt 높음",
     "Vt가 높고 on 전류가 크게 줄었는데, silicide를 막은 poly 저항도 커졌어요. SAB 패턴이 있어야 poly 도핑 부족이 "
     "면저항으로 보여요 — silicide가 덮인 poly로는 안 보여요."),
    ("N4", "n26_rta_edge", "가장자리 링 + S/D 면저항 — 몇 °C",
     "가장자리 칩들의 S/D 면저항이 커요(silicide를 막은 시험 패턴). 가장자리 링 후보 9개 중 RTA 하나로 좁히고, "
     "면저항 차이로 가장자리가 몇 °C 덜 데워졌는지까지 계산해요."),
    ("N5", "n30_contact_barc", "가장자리 일부 + 연결 끊김",
     "한쪽 가장자리 칩만 끊겼어요. 가장자리 일부 후보는 둘인데 바뀌는 측정이 달라 갈려요."),
    ("N6", "n21b_develop_residue", "도넛 + 배선 누설 · 끊김",
     "고리 반경의 칩들이 배선 누설과 끊김으로 떨어졌어요. 도넛을 만드는 단계는 gate 포토 현상 하나예요."),
    ("N7", "n29_ild_cmp_edge", "가장자리 링 + 연결 끊김 — 안 갈리는 짝",
     "가장자리 칩이 끊겼어요. ILD CMP와 contact 식각은 모양도 측정도 같아서, 엔진은 둘을 같이 올리고 무엇을 재면 "
     "갈리는지 말해요."),
    ("N8", "n18_rinse_charge", "가운데 + gate 누설 — 셋이 묶임",
     "가운데 칩의 gate가 샜어요. 세정 헹굼 · gate 뒤 ashing · contact 뒤 ashing 셋이 같은 모양 · 같은 측정이에요."),
    ("N10", "n24_sd_dose", "거의 전체 + S/D 면저항 — 도즈",
     "웨이퍼 전체의 S/D 면저항이 커졌어요. 같은 면저항이라도 N4처럼 가장자리 링이면 RTA, 거의 전체면 S/D 주입 도즈예요 — "
     "측정이 같으면 맵 모양이 가려요."),
    ("N11", "n27_salicide_temp", "거의 전체 + silicide 면저항 — 고저항 상",
     "silicide가 덮인 패턴의 면저항이 두 배 가까이 됐어요. salicide 열처리 온도가 낮아 고저항 상(C49)이 얼마나 남았는지 "
     "계산해요."),
    ("N9", "n19_gox_thick", "거의 전체 + Vt 높음 · on 전류 부족 — 얼마나",
     "Vt가 높고 on 전류도 줄었어요. 단계는 gate 산화와 screen oxide 잔막이 공동이지만, C-V로 잰 산화막 두께로 "
     "「얼마나」는 둘 다 같이 말할 수 있어요."),
    # 2026-10-28 추가 — 근거 지도(wiki 원인공정추천-NMOS-근거지도)의 후보 ③ · ④ · ⑥ · ①
    ("N12", "n10_fill_void", "한 곳 뭉침 + 분리 누설 — 칩당 결함 수",
     "한 자리에 뭉친 칩들이 이웃 소자 사이로 샜어요(분리 누설). 장비에서 떨어진 입자가 한 곳에 뭉치는 건 원문(Hansen · "
     "Stapper)이고, 그 장비가 STI를 채우는 단계라는 건 우리 배정이에요. 맵의 불량 칩 비율로 그 자리의 칩당 결함 수를 계산해요."),
    ("N13", "n37_forming_gas_skip", "거의 전체 + Vt · SS — 사고 둘이 겹침",
     "Vt 주입 도즈가 많고 forming gas도 빼먹은 웨이퍼예요. 파라미터 하나로는 WAT 값이 설명되지 않아서, 엔진이 "
     "「사고가 겹쳤을 가능성」을 경고하는지 봐요."),
    ("N14", "n07_trench_edge", "가장자리 링 + 분리 누설 — trench 식각",
     "가장자리 칩들이 이웃 소자 사이로 샜어요. 가장자리 링 후보 9개 중 분리 누설은 trench 식각 하나예요. edge ring 원문"
     "(Chai · Brady)을 trench 식각으로 옮긴 경우라 근거는 「옮김」이에요."),
    ("N15", "n22_poly_overetch", "가장자리 링 + gate 누설 — poly 식각",
     "같은 가장자리 링인데 이번엔 gate가 샜어요. 가장자리에서 poly를 너무 깎아 gate 산화막이 다친 경우예요(옮김)."),
    ("N16", "n34_w_cmp_edge", "가장자리 링 + 배선 누설 — W CMP",
     "가장자리 칩들의 배선이 붙었어요. 가장자리에서 W가 덜 깎여 남은 경우예요(Goh의 CMP 가장자리 두께를 옮김)."),
    ("N17", "n38_probe", "거의 전체 + 연결 끊김 — 공정이 아니라 검사?",
     "거의 전체 칩이 연결 끊김으로 떨어졌어요. 검사 바늘이 덜 닿았다면 공정은 멀쩡해요 — pad가 덜 열린 경우와 똑같이 "
     "보여서, 다시 재 보면 갈려요(IBM · TI 특허)."),
]
# 사고를 둘 넣는 시나리오 — 칩 측정은 두 원인의 예상 비율을 반씩 섞고, 정답은 둘 다
MIX = {"N13": ["n16_vt_dose", "n37_forming_gas_skip"]}
# 「얼마나」를 보일 시나리오의 사고 크기 — 정상 산포로 뽑은 웨이퍼에 얹는다 (나머지 시나리오의 WAT 값은 정상 웨이퍼)
PLANT = {"N1": ("Na", lambda p: p["Na"] * 1.45),        # 채널 도핑 +45% → Vt 약 +0.14 V
         "N2": ("Dit", lambda p: p["Dit"] * 100),       # forming gas 생략 — 계면 트랩 약 100배 (Mack 2023)
         "N3": ("Ndp", lambda p: 5e18),                 # poly 도핑 10²⁰ → 5×10¹⁸
         "N9": ("tox", lambda p: 75e-8),                # gate 산화막 60 → 75 Å
         "N4": ("dT", lambda p: p["dT"] - 20.0),        # RTA 가장자리 −20 °C (가장자리 site에서 잰다)
         "N10": ("sd", lambda p: p["sd"] * 0.65),       # S/D 활성 도즈 −35%
         "N11": ("c49", lambda p: 0.35),                # 고저항 상 C49가 35% 남음
         "N13": [("Na", lambda p: p["Na"] * 1.45),      # 둘 다 — Vt 주입 도즈 +45%
                 ("Dit", lambda p: p["Dit"] * 100)]}    #        + forming gas 생략


def card(c):
    suspect, shape, checks = TEXT[c.id]
    labels = " · ".join(nk.BIN_LABELS[b] for b in c.primary)
    step_txt = " · ".join(f"{s.lstrip('0')}. {nk.STEPS[s]}" for s in c.steps) or "전 공정 공통"
    return {"title": c.name, "suspect": suspect, "this": f"{c.name} — {c.knob}",
            "normal": f"{labels} 칩이 거의 없음", "checks": checks,
            "what": f"{step_txt} — {c.knob}",
            "shape": f"{PAT_KO[c.pattern]} — {shape or SHAPE_DEFAULT[c.pattern]}",
            "test": f"{labels} 쪽으로 쏠려요 — " + " · ".join(BIN_PLAIN_TEST[b] for b in c.primary),
            "source": f"근거 수준 「{c.evidence}」 · {c.source}", "evidence": c.evidence,
            "kind": KIND[c.id], "kind_note": kind_note(c, checks)}


def density(wm, rng):
    """결함 밀도 유형의 「얼마나」 — 실제 맵의 불량 칩 비율을 Stapper 1983 p.458 식 (15)(Poisson 수율 Y = e^−λ)로 거꾸로
    풀어 칩당 결함 수 λ를 낸다. 가장 큰 불량 덩어리(4방향 연결)를 한 칸 넓힌 자리와 나머지를 따로 보고, 90% 범위는
    합격 칩 수의 베타 분포에서. 군집을 넣은 식(p.460 식 (25))은 군집 계수를 몰라 쓰지 않는다."""
    fail, on = wm == 2, wm > 0
    lab = np.zeros(wm.shape, int)
    best, best_n, cur = 0, 0, 0
    for i, j in zip(*np.where(fail)):
        if lab[i, j]:
            continue
        cur += 1
        stack, n = [(i, j)], 0
        lab[i, j] = cur
        while stack:
            a, b = stack.pop()
            n += 1
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                x, z = a + da, b + db
                if 0 <= x < wm.shape[0] and 0 <= z < wm.shape[1] and fail[x, z] and not lab[x, z]:
                    lab[x, z] = cur
                    stack.append((x, z))
        if n > best_n:
            best, best_n = cur, n
    core = lab == best
    region = core.copy()
    region[1:, :] |= core[:-1, :]
    region[:-1, :] |= core[1:, :]
    region[:, 1:] |= core[:, :-1]
    region[:, :-1] |= core[:, 1:]
    region &= on

    def lam(mask):
        n, k = int(mask.sum()), int((mask & (wm == 1)).sum())
        return np.percentile(-np.log(rng.beta(k + 1, n - k + 1, 4000)), [5, 50, 95]), n

    (l5, l50, l95), n_in = lam(region)
    (_, r50, _), _ = lam(on & ~region)
    return {"param": "lam", "kind": "density", "label": "그 자리 칩당 결함 수", "from": "불량 칩 비율 · Stapper 수율 식",
            "value": f"{l50:.1f}개", "lo": f"{l5:.1f}개", "hi": f"{l95:.1f}개", "normal": f"나머지 {r50:.2f}개",
            "region": n_in, "out_of_range": bool(l50 > 3 * r50), "consistent": True, "mismatch": []}


def rank(P, counts, signatures=None):
    """순위 — n1_nmos_identify.judge와 같은 판정(모양 후보 밖은 확률 문턱을 넘어야 남는다)."""
    sc = nk.score(P[None], counts[None], signatures=signatures)[0]
    post = kb.softmax(sc[None])[0]
    elig = nk.eligible(post[None], np.array([1 + int(P[1:].argmax())]))[0]
    order = [int(i) for i in np.argsort(-np.where(elig, post, -1), kind="stable") if elig[i]]
    top = [i for i in order if post[i] >= post[order[0]] * (1 - kb.TIE_RTOL)]
    return order, top, post


def main():
    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maps_u8, y, lots, lot_names = load_cache()
    _, _, m_te = split_by_lot(lots)
    te = np.where(m_te)[0]
    P0 = cnn_probs(maps_u8[te], device)
    pred, conf = P0.argmax(1), P0.max(1)
    used, out = set(), []
    for n_s, (sid, cid, title, story) in enumerate(SCENARIOS):
        rng = np.random.default_rng(SEED + n_s)
        c = nk.CAUSES[nk.CAUSE_INDEX[cid]]
        pi = kb.PATTERNS.index(c.pattern)
        cand = np.where((y[te] == pi) & (pred == pi) & (conf >= 0.90))[0]
        # 모양이 또렷한 맵 — make_moscap_scenarios와 같은 고르기(결함 다이 수가 중앙값에 가까운 맵, lot 겹치지 않게)
        n_def = (maps_u8[te[cand]] == 2).sum(axis=(1, 2))
        order = cand[np.lexsort((-conf[cand], np.abs(n_def - np.median(n_def))))]
        pick = next(int(k) for k in order if int(lots[te[k]]) not in used)
        gi = int(te[pick])
        used.add(int(lots[gi]))
        wm = maps_u8[gi]
        fail = wm == 2
        sig = np.mean([nk.SIGNATURES[nk.CAUSE_INDEX[x]] for x in MIX.get(sid, [cid])], axis=0)
        b = rng.choice(nk.B, size=int(fail.sum()), p=sig / sig.sum())
        flip = rng.random(b.size) < EPS_BIN
        b[flip] = rng.integers(0, nk.B, int(flip.sum()))
        die = wm.astype(int)
        die[fail] = kb.BIN_CODE_OFFSET + b
        counts = np.bincount(b, minlength=nk.B).astype(float)
        P = P0[pick]
        order_c, top, post = rank(P, counts)
        _, top4, _ = rank(P, nk.collapse4(counts[None])[0], signatures=nk.SIGNATURES4)
        keep = order_c[:max(3, len(top) + 2)]
        ranking = [{"cause_id": nk.CAUSE_IDS[i], "steps": list(nk.CAUSES[i].steps),
                    "post_pct": round(float(post[i]) * 100, 1), "tied": len(top) > 1 and i in top} for i in keep]
        top_ids = frozenset(nk.CAUSE_IDS[i] for i in top)
        if len(top) == 1:
            note = ("good", f"측정 {nk.B}칸으로 단독 1순위예요. 같은 측정을 기존 4칸(끊김 · 누설 · 문턱 이탈 · 전류 부족)으로 "
                            + (f"묶었다면 공동 1순위가 {len(top4)}개였어요." if len(top4) > 1 else "묶어도 단독이에요."))
        else:
            split = SPLIT.get(top_ids)
            note = ("info", f"측정 {nk.B}칸으로도 안 갈리는 {len(top)}개 묶음이에요"
                            + (f" — {split}을 재면 갈려요." if split else ".")
                            + f" 기존 4칸이었다면 공동 {len(top4)}개였어요.")
        # WAT 값 — 정상 웨이퍼 하나를 뽑고, 사고 크기를 얹는다
        rng_w = np.random.default_rng(SEED + 100 + n_s)
        params = {k: float(v[0]) for k, v in nd.sample_normal(rng_w, 1).items()}
        planted = None
        if sid in PLANT:
            pl = PLANT[sid] if isinstance(PLANT[sid], list) else [PLANT[sid]]
            for prm, f in pl:
                params[prm] = float(f(params))
            if len(pl) == 1:
                planted = {"param": prm, "label": nd.KNOB[prm][0], "value": nd.fmt(prm, params[prm]),
                           "normal": nd.fmt_range(prm, *nd.normal_range(prm)), "raw": params[prm]}
            else:
                planted = {"param": "+".join(q for q, _ in pl), "label": " + ".join(nd.KNOB[q][0] for q, _ in pl),
                           "value": " · ".join(nd.fmt(q, params[q]) for q, _ in pl), "normal": "—", "multi": True}
        meas = nd.measure(params, rng_w)
        if planted and not planted.get("multi"):
            # 넣은 값이 90% 범위 밖이면 왜 그런지 — 정상 산포에서 가장 멀리 뽑힌 다른 파라미터
            raw = nd.estimate(cid, meas)["raw"]
            # 화면 자릿수로 같으면 안으로 본다 — 「5.0 vs 5.0 ~ 5.2」를 밖이라고 하면 보는 사람이 헷갈린다
            planted["inside"] = bool(raw["p5"] <= planted["raw"] <= raw["p95"] or nd.fmt(prm, planted["raw"]) in
                                     (nd.fmt(prm, raw["p5"]), nd.fmt(prm, raw["p95"])))
            z = {k: (np.log(params[k] / nd.NOM[k]) if k == "Dit" else
                     (params[k] - nd.NOM[k]) if k in ("dT", "c49") else (params[k] / nd.NOM[k] - 1)) / nd.SPREAD[k]
                 for k in nd.NOM if k != planted["param"]}
            kz = max(z, key=lambda k: abs(z[k]))
            planted["nuisance"], planted["nuisance_z"] = nd.KNOB[kz][0], float(abs(z[kz]))
        mag = {}
        dens = None
        for r in ranking:
            e = nd.estimate(r["cause_id"], meas)
            if e is not None:
                mag[r["cause_id"]] = {k: v for k, v in e.items() if k != "raw"}
            elif KIND[r["cause_id"]] == "결함 밀도":      # 크기는 원인이 아니라 맵에서 — 결함 밀도 원인끼리 같은 값
                dens = dens or density(wm, np.random.default_rng(SEED + 200 + n_s))
                mag[r["cause_id"]] = dens
        # 맵 모양(CNN 1위)이 남긴 후보의 파라미터만 — 막은 S/D 면저항 하나로는 도즈와 RTA 온도가 같아 보인다
        fp = [{k: v for k, v in r.items() if k != "raw"} for r in
              nd.fingerprint(meas, causes=nk.PATTERN_CANDIDATES[kb.PATTERNS[1 + int(P[1:].argmax())]])]
        # WAT가 합격선 밖인데 어느 파라미터 하나로도 설명이 안 되면 — 사고가 겹쳤을 가능성 (「얼마나」를 믿지 않는다)
        wat_bad = [nd.MEAS_LABEL[k] for k, v in nd.in_spec(meas).items() if not v]
        warning = None
        if wat_bad and fp and not any(r["explains"] for r in fp):
            warning = (f"WAT 값({' · '.join(wat_bad)})이 합격선 밖인데, 파라미터 하나로는 설명이 안 돼요 — 사고가 둘 이상 "
                       "겹쳤을 가능성이 있어요. 1순위의 「얼마나」는 믿지 말고, 후보 단계들의 기록을 같이 확인하세요.")
        truth = MIX.get(sid, [cid])
        top3_pat = np.argsort(-P)[:3]
        item = {
            "id": sid, "title": title, "story": story, "set_cause": cid, "truth": truth, "warning": warning,
            "source": {"dataset": "WM-811K", "split": "A-1b lot-wise test", "lot": str(lot_names[int(lots[gi])]),
                       "cache_index": gi, "true_pattern": c.pattern},
            "cnn": {"top": kb.PATTERNS[int(P.argmax())], "conf_pct": round(float(P.max()) * 100, 1),
                    "top3": [{"pattern": kb.PATTERNS[int(i)], "pct": round(float(P[i]) * 100, 1)} for i in top3_pat]},
            "bin_counts": dict(zip(nk.BIN_TYPES, counts.astype(int).tolist())),
            "bins_used": bool(nk.informative(counts[None])[0]),
            "die_map": die.tolist(), "ranking": ranking, "tie_size": len(top), "tie_size_4bin": len(top4),
            "note": note,
            "wat": {"meas": meas, "in_spec": {k: bool(v) for k, v in nd.in_spec(meas).items()},
                    "where": ("가장자리 site 4곳 중앙값" if c.pattern in ("edge-ring", "edge-loc") else "웨이퍼 9곳 중앙값")},
            "planted": planted, "magnitude": mag, "fingerprint": fp,
        }
        out.append(item)
        hit = any(t in top_ids for t in truth)
        print(f"{sid} {title:<34} CNN {item['cnn']['top']:<9} {item['cnn']['conf_pct']:5.1f}%  "
              f"1위 {'·'.join(sorted(top_ids)):<60} {'✓' if hit else '✗'}  {nk.B}칸 공동 {len(top)} · 4칸 공동 {len(top4)}")
        if warning:
            print("     ⚠️", warning)
        if dens:
            print(f"     칩당 결함 수 — 그 자리 {dens['value']} ({dens['lo']} ~ {dens['hi']}, {dens['region']}칸) · {dens['normal']}")
        if planted and not planted.get("multi"):
            e = mag.get(cid)
            print(f"     넣은 크기 {planted['label']} {planted['value']} → 엔진 {e['value']} ({e['lo']} ~ {e['hi']}) "
                  f"· 정상 {e['normal']} · 다른 측정과 {'맞음' if e['consistent'] else '안 맞음 ' + str(e['mismatch'])} · "
                  f"설명하는 줄 {[r['label'] for r in item['fingerprint'] if r['explains']]}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        # 대시보드는 src.inference를 import하지 않는다 — 화면 문구를 같이 담는다
        json.dump({"flow": [[k, n] for k, n in nk.STEPS.items()], "groups": GROUPS, "bin_types": nk.BIN_TYPES,
                   "bin_plain": nk.BIN_LABELS, "bin_plain_test": BIN_PLAIN_TEST, "colors": COLORS,
                   "card": {c.id: card(c) for c in nk.CAUSES}, "unobservable": nk.UNOBSERVABLE,
                   "wat_spec": {"base": nd.BASE, "spec": nd.SPEC, "label": nd.MEAS_LABEL,
                                "tox_range": [x * 1e8 for x in nd.normal_range("tox")],
                                "normal_range": {prm: nd.fmt_range(prm, *nd.normal_range(prm)) for prm in nd.KNOB}},
                   "scenarios": out}, f, ensure_ascii=False)
    print("저장:", OUT)


if __name__ == "__main__":
    main()
