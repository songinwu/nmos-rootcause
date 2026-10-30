# NMOS 원인 공정 추천 — 지식표 · 「얼마나」 · 합성 시험 · 시연 화면

종합설계 「웨이퍼 불량 지도로 원인 공정 좁히기」에서 **NMOS 부분만** 뽑아 온 저장소예요.
엔진 본체 · 대시보드는 비공개 저장소에 있어서, 원인 표 쪽 파일은 **읽기용**이고 혼자서는 실행되지 않아요.
**모양 CNN 학습 코드는 여기서 그대로 돌아가요** — 아래 [모양 CNN](#모양-cnn--학습-코드-2026-10-30)을 보세요.

> **최신 — 2026-10-30** · **fail 라벨을 문헌대로 다시 붙였어요** — 원인 40 → 39, 주 fail 50%, 시험용 lot 맵만으로 다시 시험 ([결과 1](#결과-1--어느-단계를-바로-짚나-합성-시험-n-1) · [`docs/relabel_plan.md`](docs/relabel_plan.md)) · 모양 CNN 학습 코드 · 가중치 추가.
> **2026-10-28** · 41단계(38 + SAB 3) · 측정 12칸 · 「얼마나」를 말하는 원인 8개 + 결함 밀도 · 시연 웨이퍼 17장.
> `docs/`의 조원 설명 PDF는 **3판(2026-10-30 — 41단계 · 측정 12칸 · 원인 39개)**이고, 파라미터 범위 근거 PDF도 같이 있어요.

## 무엇을 했나

1. **공정을 정했어요.** NMOS 하나를 41단계 · 마스크 6장으로 만들어요. IDEC 0.5 µm 공정에서 NMOS 부분만 가져오고,
   소자 분리는 STI(KSIA)로, LDD는 빼고, silicide를 막는 층(SAB)을 넣었어요.
2. **원인 표를 만들었어요.** 단계마다 「틀어지면 **웨이퍼맵이 어떤 모양**이 되고 **칩 측정 중 무엇이 바뀌나**」를 원인 39개로 적었어요.
3. **「어느 단계」** — 엔진은 맵 모양으로 후보를 좁히고, 칩 측정 12칸으로 그중에서 골라요.
4. **「얼마나」** — 웨이퍼 WAT 값(Vt · SS · on 전류 · C-V 산화막 · 면저항 셋)을 MOS 식으로 거꾸로 풀어
   「gate 산화막 약 75 Å, 정상 53 ~ 68 Å 밖」처럼 파라미터가 얼마나 벗어났는지 말해요.

**EDS 맵이 「어느 단계」를, WAT 값이 「얼마나」를 맡아요.** 장비 설정(산화로 몇 °C · 몇 분)은 같은 결과를 만드는 조합이
여러 개라 정하지 않고 장비 기록으로 넘겨요. 예외는 RTA예요 — 면저항과 온도의 관계를 공장이 실제로 이렇게 써요(Riley 1996).

## 먼저 읽을 것

1. 이 README
2. [`src/inference/rootcause_kb_nmos.py`](src/inference/rootcause_kb_nmos.py) — 41단계(`STEPS`)와 원인 39개(`CAUSES`) · fail 근거(`FAIL_EVIDENCE`)
3. [`src/inference/nmos_device.py`](src/inference/nmos_device.py) — 「얼마나」 식 · 기준값 · 합격선 · 불량 경계 (맨 위 설명에 출처)
4. [`docs/NMOS_원인공정_지식표_조원설명.pdf`](docs/NMOS_원인공정_지식표_조원설명.pdf) — 41단계 · 측정 12칸 · 연결표 · fail 근거 · 시험 결과 · 시연 17장 (3판)
5. [`docs/NMOS_파라미터범위_근거.pdf`](docs/NMOS_파라미터범위_근거.pdf) — 파라미터 범위 7개를 어떻게 정했나 · 기준값 · 합격선 · 식의 출처

## 파일

| 파일 | 내용 |
|---|---|
| `src/inference/rootcause_kb_nmos.py` | 지식표 — 41단계, 측정 12칸, 원인 39개(단계 · 무엇이 틀어지나 · 바뀌는 측정 · 맵 모양 · 근거) · fail 근거 표시 · 라벨 비율(주 fail 50%) |
| `src/inference/nmos_device.py` | 「얼마나」 — 공정값 ↔ WAT 값 식, 기준값 · 합격선 · 불량 경계, 역산(중앙값 · 90% 범위 · 다른 측정과 맞나) |
| `scripts/n1_nmos_identify.py` | 합성 시험 N-1 — 원인마다 **시험용 lot** 맵 200장에 사고를 심고 엔진이 되찾는지 봄 (주 fail 30 · 50 · 70%) |
| `results/n1_nmos_relabel.json` · `results/n1_nmos_1028.json` | N-1 결과 — 다시 붙인 라벨(10-30) · 예전 라벨(10-28) |
| `scripts/make_nmos_scenarios.py` | 시연 웨이퍼 17장(N1 ~ N17) — 맵 · CNN 판정 · 칩 측정 · WAT 값 · 크기 · 경고 · 화면 문구 |
| `data/demo/nmos_scenarios.json` | 위 스크립트가 만든 결과 |
| `src/dashboard/demo16.py` | 대시보드 섹션 16 시연 화면 (MOSCAP · CMOS · NMOS) |
| `tests/test_rootcause_nmos.py` · `tests/test_nmos_device.py` · `tests/test_demo16_text.py` | 표 모양 · 대표 판정 · 크기 역산 · 시나리오가 엔진과 맞는지 |
| `docs/…조원설명.pdf` | 조원 설명용 정리본 (3판, 19쪽) |
| `docs/NMOS_파라미터범위_근거.pdf` | 파라미터 범위 근거 (11쪽) |
| `docs/relabel_plan.md` | fail 라벨 다시 붙이기 — 예전 방식 · 바꾼 것 · 데이터 나누기 · 문헌 말 번역 규칙 · leakage 논문 4편 · 원인별 fail 근거 (2026-10-30) |
| `src/training/train_wafermap.py` | 모양 CNN — 모델 구조 · WM-811K 읽기(26×26) · 웨이퍼 단위 분할 학습 |
| `scripts/a1b_lotwise_model.py` · `scripts/a1_lot_consistency.py` | 모양 CNN — **lot 단위 분할 학습(시연 · 엔진이 쓰는 것)** · lot 뭉침 검정 |
| `src/inference/wafermap_predictor.py` | 학습한 CNN으로 맵 한 장 판정 |
| `scripts/download_wm811k.sh` · `requirements-cnn.txt` | WM-811K 받기(Kaggle) · 필요한 패키지 |
| `results/cnn_lotwise.json` · `results/cnn_waferwise.json` | 두 학습의 결과 숫자 |
| `models/wafermap/wafermap_cnn_lotwise.pt` · `wafermap_cnn.pt` | 학습한 가중치 (각 2.5 MB) — lot 단위 판 · 웨이퍼 단위 판 |

엔진 본체(`rootcause_kb.score_batch`)는 이 저장소에 없어요.
엔진이 하는 일은 「모양으로 고른 후보 중에서, 칩 측정이 그 원인의 예상과 가장 잘 맞는 순서로 줄 세우기」예요.

## 결과 1 — 「어느 단계」를 바로 짚나 (합성 시험 N-1)

> ⚠️ **아래 숫자는 상한이에요.** 사고를 심을 때와 채점할 때 같은 원인 표를 써서 「표가 맞다면 이만큼」이에요.
> 표와 따로 만든 문제로 채점한 시험은 NMOS에는 아직 없어요(아래 「한계」).

*(2026-10-30 다시 함)* 원인마다 그 모양의 **시험용 lot**(CNN이 한 번도 안 본 lot) 맵 200장에 사고를 심었어요.
불량 칩마다 주 fail 50%(둘이면 25 + 25) · 나머지 50%는 다른 칸에 고르게 라벨을 붙여요.
시험용 맵이 200장보다 적은 모양(거의 전체 19장 · 도넛 71장 · 흩어짐 114장 · 긁힘 181장)은 90° 회전 4방향 × 좌우 뒤집기로 8배 늘려 그중에서 뽑았어요.
엔진이 그 원인을 **단독 1위**(동점 없이 1위)로 되찾는 비율이 90% 이상이면 「바로 짚음」으로 셌어요.

| 조건 | 측정 12칸 | 기존 4칸 | 정답이 공동 1위 안 |
|---|---|---|---|
| 맵 모양을 정답으로 줬을 때 | **21개** | 7개 | 100% |
| 모양 CNN을 거쳤을 때 | **20개** | 5개 | 97.7% |
| *(10-28 예전 라벨 · 학습 맵 섞임)* | *23 · 23개* | *10 · 9개* | *99.96 · 97.5%* |

- 표만 보고 「단독일 것」이라고 한 예측과 39개 모두 일치했어요. 나머지 18개는 2 ~ 4개 묶음이에요 — 모양도 측정도 원리상 같아서
  공정 중 측정이나 장비 기록으로 갈라요.
- 23 → 21개는 원인 표를 고친 결과예요. 원문대로 고쳤더니 묶음이 둘 생겼어요(05 · 21a 노광 초점 둘, 08 · 10 STI 쪽 둘).
- CNN을 거치면 21 → 20개 — 28 ILD 입자(한 곳 뭉침)가 74%로 떨어졌어요. 처음 보는 lot에서 CNN이 「한 곳 뭉침」을 맞히는 비율이 71%라서예요.
  예전 23개는 CNN이 학습한 맵까지 섞여 있어 낙관적이었어요.

**주 fail 몫이 틀리면** — 50%는 근거 없는 설계값이라, 라벨만 30 · 70%로 붙이고 엔진은 50% 표로 채점했어요.

| 라벨의 주 fail 몫 | 참 모양 | 모양 CNN |
|---|---|---|
| 30% (엔진 생각보다 흐림) | 17개 | 15개 |
| **50%** | **21개** | **20개** |
| 70% (엔진 생각보다 또렷함) | 21개 | 20개 |

30%에서 떨어지는 것은 주 fail이 둘 · 셋인 원인(01 · 20)과 CNN이 약한 한 곳 뭉침(13 · 28)이에요.

**비교 기준 — 같은 시험에서 넣는 것을 줄이면** *(2026-10-30)*

| 엔진에 넣은 것 | 바로 짚는 원인 (단독 1위 90% 이상) | 평균 단독 1위 | 정답이 3순위 안 |
|---|---|---|---|
| 무작위로 하나 고르기 | 0개 | 2.6% | 7.7% |
| 모양만 (CNN) | 2개 | 4.7% | 38.2% |
| 모양 + 측정 4칸 (CNN) | 5개 | 18.1% | 83.3% |
| **모양 + 측정 12칸 (CNN)** | **20개** | **52.1%** | **96.2%** |

- 모양만으로는 원인 39개가 모양 9가지로만 갈려 거의 못 짚어요(맵 모양을 정답으로 줘도 2개).
  측정 결과를 붙여야 단계가 갈리고, 칸을 잘게 나눌수록(4 → 12칸) 더 갈려요.
- 넷 다 같은 표로 심고 채점한 상한이라 **서로 견주는 데만** 써요.

## 결과 2 — 「얼마나」

**기준값과 합격선** (출처는 `nmos_device.py` 맨 위)

| 항목 | 값 | 출처 |
|---|---|---|
| gate 산화막 | 60 Å | IDEC 0.5 µm 강의 |
| Vt 기준 · 합격선 | 0.50 V · ±0.10 V | 기준은 MOSIS TSMC 0.25 µm 실측(산화막 57 Å · Vt 0.49 ~ 0.50 V), 폭은 GF180MCU 3.3 V NMOS 공개 spec |
| on 전류 합격선 | ±16% | GF180MCU 공개 spec 폭 |
| 면저항 기준 | 막은 S/D 58 · 막은 poly 176 · silicide 덮인 곳 3.5 Ω/sq | MOSIS T7CZ 실측 |
| RTA 온도 ↔ 면저항 | 0.5 Ω/sq per °C | Riley 1996 p.20 (비소 · 1,025 °C를 옮김 — 근사) |
| SS 합격선 · 면저항 합격선 | 기준 +10 mV/dec · ±15% | 가정 |

**불량 경계** (파라미터 하나만 움직였을 때 측정이 전부 합격선 안인 범위)
— 산화막 53 ~ 68 Å · 채널 도핑 −27 ~ +31% · poly 도핑 8.7×10¹⁹ ~ 1.2×10²⁰ cm⁻³ · 계면 트랩 기준의 60배 이하 ·
S/D 활성 도즈 −13 ~ +18% · RTA 온도 −17 ~ +17 °C · 남은 고저항 상(C49) 6% 이하

**예: Vt가 0.50 → 0.65 V일 때, 후보 하나가 혼자서 이걸 만들려면** (장채널 MOS 식 — 교과서)

| 후보 단계 · 파라미터 | 필요한 이탈 | SS | on 전류 | 산화막 (C-V) |
|---|---|---|---|---|
| 19 gate 산화 · 산화막 두께 | 60 → 75 Å (+25%) | +5 | −26% | **75 Å** |
| 16 Vt 주입 · 채널 도핑 | +48% | +4 | −10% | 60 Å |
| 37 forming gas · 계면 트랩 | ×120 | **+20** | −9% | 60 Å |
| 20 poly 증착 · poly 도핑 | 10²⁰ → 3.6×10¹⁸ | +6 | **−60%** | 60 Å |

Vt 하나로는 넷 다 가능하지만, 함께 바뀌는 측정이 달라서 갈려요. 엔진은 파라미터를 하나씩 의심해 보고
**「정상 범위 밖이고, 그 크기면 나와야 할 다른 측정도 이번과 맞는」** 줄을 이번 일의 설명으로 골라요.

**얼마나 정확한가** — 원인 7개 × 웨이퍼 120장: 90% 범위가 넣은 값을 담는 비율 **88 ~ 94%**,
「설명하는 줄이 정답 하나」 98 ~ 100%.

**사고가 둘 겹치면** — WAT 값이 합격선 밖인데 어느 파라미터 하나로도 설명이 안 되면, 엔진은 크기를 말하는 대신
「사고가 겹쳤을 가능성」을 경고해요(N13). 사고를 넣지 않은 웨이퍼에는 경고가 뜨지 않는 것을 테스트로 확인했어요.

| 유형 | 원인 수 | 크기까지 말하나 |
|---|---|---|
| 전기 값 | 11 | ✅ WAT 값을 식으로 역산 — 식을 붙인 것 8개(16 · 17 · 19 · 20 · 24 · 26 · 27 · 37) |
| 결함 밀도 | 6 | ✅ 칩당 결함 수 — 실제 맵의 불량 칩 비율을 Stapper 1983 Poisson 수율 식(p.458)으로 거꾸로 (N12) |
| 여유 | 16 | 🟡 「경계를 넘었다」까지 — 끊김은 예 / 아니오라 크기가 안 남음 |
| 장비 변수 | 6 | ❌ 칩 측정에 안 남음 → 장비 기록 |

## SAB 추가 (2026-10-28)

**왜** — silicide가 S/D와 poly를 덮으면 면저항이 17 ~ 57배 작아져서(MOSIS 실측 N+ 3.5 vs 막은 곳 58 Ω/sq),
S/D 도즈가 30% 줄어도 면저항은 1.7%밖에 안 변해요(병렬 저항 계산). 공장은 silicide를 막은 시험 패턴으로 재요.

**무엇이 바뀌었나**

- 활성화 RTA(26) 뒤에 **26a SAB 산화막 증착 · 26b SAB 포토 · 26c SAB 식각 · PR 제거**를 끼웠어요. 기존 번호를 지키려고 a · b · c를 붙였어요.
- 면저항을 패턴 셋으로 나눠 **측정 12칸**이 됐어요 (막은 S/D · 막은 poly · silicide 덮인 곳).
- 원인 넷의 측정 방향을 원문대로 고쳤어요.

| 원인 | 전 | 후 | 근거 |
|---|---|---|---|
| 24 S/D 주입 | S/D 면저항 · on 전류 | 막은 S/D 면저항 | 0.5 µm면 S/D 직렬 저항이 채널 저항보다 훨씬 작아 on 전류엔 거의 안 보임 (원리 계산) |
| 26 RTA 가장자리 | S/D 면저항 · on 전류 | 막은 S/D 면저항 | 공장도 막은 면저항으로 RTA 온도를 판정 (Riley 1996) |
| 27 salicide 온도 | S/D · poly 면저항 | silicide 면저항 | 고저항 상 C49 60 ~ 70 vs C54 15 ~ 20 µΩ·cm (Clevenger 1994) |
| 20 poly 도핑 | poly 면저항 · Vt | 막은 poly 면저항 · Vt · on 전류 | 덮인 poly로는 안 보임 (Clevenger 1994) · poly 공핍 |

- 막은 S/D 면저항 하나로는 도즈 부족과 RTA 온도 부족이 같아 보여요. 둘은 맵 모양(거의 전체 vs 가장자리 링)이 갈라요.

## 시연 화면 N1 ~ N17

| 웨이퍼 | 모양 + 측정 | 엔진 1순위 | 얼마나 (넣은 → 엔진) |
|---|---|---|---|
| N1 | 거의 전체 + Vt 높음 | 16 Vt 주입 | 채널 도핑 +44% → +43% |
| N2 | 거의 전체 + SS 큼 | 37 forming gas | 계면 트랩 ×116 → ×119 |
| N3 | 거의 전체 + poly 면저항 · Vt | 20 poly 도핑 | 5.0×10¹⁸ → 5.1×10¹⁸ cm⁻³ |
| N4 | 가장자리 링 + S/D 면저항 | 26 RTA | −22 °C → −20 °C ※ |
| N5 | 가장자리 일부 + 연결 끊김 | 30 contact 포토 | — (여유) |
| N6 | 도넛 + 배선 누설 | 21 gate 포토 현상 | — (장비 변수) |
| N7 | 가장자리 링 + 연결 끊김 | 29 · 31 공동 | — (ILD 두께 vs ring RF 시간으로 가름) |
| N8 | 가운데 + gate 누설 | 23 · 35 공동 (gate 뒤 · metal 1 뒤 ashing) | — (ashing 장비 기록) |
| N9 | 거의 전체 + Vt · on 전류 | 17 · 19 공동 | 산화막 75 Å → 75 Å (둘 다 같은 크기) |
| N10 | 거의 전체 + S/D 면저항 | 24 S/D 주입 | 도즈 −35% → −36% |
| N11 | 거의 전체 + silicide 면저항 | 27 salicide | 남은 C49 35% → 35% |
| N12 | 한 곳 뭉침 + 분리 누설 | 8 · 10 공동 (STI 포토 찌꺼기 · trench 채우기) | 그 자리 칩당 결함 약 0.8개 (나머지 0.13개) — 실제 맵에서 |
| N13 | 거의 전체 + Vt · SS (사고 둘) | 16 Vt 주입 · 2순위 37 forming gas | ⚠️ 「파라미터 하나로는 설명이 안 됨」 경고 |
| N14 | 가장자리 링 + 분리 누설 | 7 trench 식각 | — (여유, 근거는 옮김) |
| N15 | 가장자리 링 + gate 누설 | 22 poly 식각 | — (여유, 근거는 옮김) |
| N16 | 가장자리 링 + 배선 누설 | 34 W CMP | — (여유, 근거는 옮김) |
| N17 | 거의 전체 + 연결 끊김 | 36 pad · 38 검사 바늘 공동 | — (다시 재 보기 vs pad 현미경으로 가름) |

※ N4는 넣은 값이 90% 범위 밖이에요. 그 웨이퍼의 S/D 도즈가 정상 산포 끝(2.6σ)에서 뽑혔기 때문이고, 화면의 「정답 공개」 창에
그 이유가 같이 나와요. 90% 범위는 10번에 1번꼴로 놓쳐요.

## 모양 CNN — 학습 코드 (2026-10-30)

엔진의 첫 단계예요. 웨이퍼맵 한 장(26×26)을 보고 모양 9종(none · center · donut · edge-loc · edge-ring · loc · near-full ·
random · scratch) 중 하나로 판정하고, 그 모양이 원인 후보를 좁혀요.

- **모델** — conv 3층 + 완전 연결 2층, 파라미터 약 62만 개(`WaferMapCNN`). 입력 값은 0 = 웨이퍼 밖 · 0.5 = 정상 칩 · 1.0 = 불량 칩.
- **데이터** — WM-811K [11]의 라벨 있는 맵 172,950장. 크기가 제각각이라 26×26으로 줄여요(NEAREST). none이 85%라 학습에서는
  15,000장으로 깎고, 클래스 가중 loss · 회전 · 반전을 써요. **데이터는 저장소에 없어요** — `scripts/download_wm811k.sh`로
  Kaggle에서 받아요. **학습한 가중치는 `models/wafermap/`에 있어서** 데이터 없이 바로 판정해 볼 수 있어요
  (`torch.load(..., weights_only=True)`로 열려요 — 텐서와 숫자만 들어 있어요).

| 학습 | 나누는 법 | 고르는 기준 | 결과 | 쓰는 곳 |
|---|---|---|---|---|
| `src/training/train_wafermap.py` | 웨이퍼 단위 85 / 15 | val accuracy | val acc 0.887 · macro F1 0.837 (40,519장) | 처음 만든 판 |
| `scripts/a1b_lotwise_model.py` | **lot 단위 70 / 15 / 15** | val macro F1 | **처음 보는 lot 1,615개 · 26,180장에서 acc 0.907 · macro F1 0.741** | 시연 · 엔진 |

⚠️ 웨이퍼 단위로 나누면 같은 lot의 웨이퍼가 학습과 검증에 섞여 숫자가 좋게 나와요. 그래서 엔진은 lot 단위로 나눈 판을 써요.
두 판은 나누는 법 · 고르는 기준 · 평가 셋이 함께 달라서 숫자를 바로 비교하면 안 돼요.

```bash
pip install -r requirements-cnn.txt
bash scripts/download_wm811k.sh            # Kaggle API 키 필요 → data/raw/wm811k/WM811K.pkl
python -m src.training.train_wafermap       # 웨이퍼 단위 판 → models/wafermap/wafermap_cnn.pt
python scripts/a1b_lotwise_model.py         # lot 단위 판 → models/wafermap/wafermap_cnn_lotwise.pt
```

```python
from src.inference.wafermap_predictor import WaferMapPredictor
p = WaferMapPredictor("models/wafermap/wafermap_cnn_lotwise.pt")
p.predict(wafer_map)   # 26×26, 값 0 / 0.5 / 1.0 → 모양 · 확신도
```

**모양별로 보면 — 정상 웨이퍼를 불량으로 부르는 오경보가 커요** *(2026-10-30, `scripts/a1d_cnn_threshold.py`)*

전체 정확도 0.907은 정상 웨이퍼(85%)가 끌어올린 숫자예요. 모양별로 「CNN이 이 모양이라 한 것 중 맞은 비율」(precision)을 보면
긁힘 13% · 한 곳 뭉침 51% · 가장자리 일부 52%예요 — 틀린 것은 대부분 정상 웨이퍼예요.
엔진은 CNN 1위가 정상이 아니면 원인을 추천하므로, 이 오경보는 그대로 「정상 웨이퍼에 원인 공정 추천」이 돼요.

그래서 **1위 모양의 확률이 그 모양의 문턱보다 낮으면 정상으로 보는 규칙**을 시험했어요. 문턱은 **검증용 lot**에서 모양마다
F1이 가장 높은 값으로 골랐고(시험용 lot은 고를 때 안 씀), 처음 보는 lot 1,615개 · 26,180장에서 쟀어요.

| 모양 | 시험용 장수 | 맞은 비율 (문턱 전 → 후) | 찾아낸 비율 (전 → 후) | 문턱 |
|---|---|---|---|---|
| 가운데 | 648 | 67% → 90% | 90% → 81% | 0.79 |
| 도넛 | 71 | 70% → 83% | 90% → 77% | 0.84 |
| 가장자리 일부 | 742 | 52% → 67% | 78% → 70% | 0.61 |
| 가장자리 링 | 1,605 | 91% → 96% | 97% → 94% | 0.79 |
| 한 곳 뭉침 | 493 | 51% → 65% | 71% → 64% | 0.53 |
| 거의 전체 | 19 | 90% → 90% | 95% → 95% | 0.00 |
| 흩어짐 | 114 | 82% → 92% | 89% → 85% | 0.69 |
| 긁힘 | 181 | 13% → 50% | 72% → 43% | 0.80 |

| | 문턱 전 | 문턱 후 |
|---|---|---|
| 정상 웨이퍼를 불량 모양으로 부름 (정상 22,307장 중) | 1,950장 | **516장** |
| 불량 웨이퍼를 정상으로 놓침 (불량 3,873장 중) | 208장 | **644장** |
| 정확도 · macro F1 | 0.907 · 0.741 | 0.951 · 0.797 |

- 오경보는 약 4분의 1로 줄지만, 놓치는 불량이 436장 늘어요(특히 긁힘은 찾아낸 비율이 반으로).
  **엔진에는 아직 붙이지 않았어요** — 오경보와 놓침 중 무엇이 더 비싼지 정한 뒤 붙여요.
- 시연 화면은 CNN이 약한 모양(맞은 비율 60% 미만)으로 판정하면 그 성적을 보이고 「추천은 후보로만 보세요」라고 경고해요.
- 신뢰구간(시험용 lot을 1,000번 다시 뽑기)은 `results/cnn_threshold.json`에 있어요.

**NMOS 맵에 쓸 때 알아 둘 것**

- WM-811K에는 **제품 종류와 웨이퍼 직경이 없어요**(컬럼 6개뿐). 웨이퍼당 칩 수가 1,266가지라 여러 제품이 섞인 양산 맵이에요.
- CNN은 **모양(위치)만** 배워요. 모양은 장비 · 공정의 위치 불균일이 만들어서 제품과 상관없이 옮겨 써요.
- **불량 칩 비율 · 칩당 결함 수는 옮기지 않아요.** 칩 설계(결함에 민감한 넓이)에 따라 같은 결함에도 죽는 비율이 달라서예요.
  시연 N12의 「칩당 결함 수」는 이 맵의 칩 기준이에요.
- 실제 NMOS 시험 웨이퍼는 불량 칩이 훨씬 적어 맵이 성글게 나올 수 있어요. 그런 맵에서는 다시 확인해야 해요.

## 근거는 어디까지 확실한가

**문헌이 받쳐 주는 것은 주로 「맵 모양」까지이고, 「어느 측정이 바뀌나」는 대부분 소자 원리로 추론한 거예요.**

| 층 | 확실한 정도 |
|---|---|
| 공정 순서 41단계 | IDEC · KSIA 강의가 근거. STI 세부(liner · 치밀화 · divot)와 SAB 위치는 일반 지식 |
| 원인 → 맵 모양 | 원인 39개 중 문헌 22개(원문 15 · 일부 3 · 옮김 4), 강의 5 · 원리 12 |
| 원인 → 바뀌는 측정(fail) | *(2026-10-30 원문 전부 다시 읽음)* 원문이 fail까지 말하는 것 **14개** · 판단 1개(01 — 「더 샌다」까지) · 원리 24개. 원인별 근거는 `rootcause_kb_nmos.py`의 `FAIL_EVIDENCE` |
| 측정이 얼마나 쏠리나 | 근거 없음 — 우리가 정한 값. *(2026-10-30)* 불량 칩마다 주 fail 50%(둘이면 25 + 25), 나머지 50%는 다른 칸에 고르게 무작위로 뽑아요. 30 · 70%로 바꿔 붙여 보는 시험은 결과 1에 있어요 (10-28판은 주 칸 하나 43% + 10% 다시 섞기) |
| 기준값 · 합격선 | 공개 실측 · 공개 spec 폭 + 일부 가정 (위 표) |

가장 단단한 것은 같은 단계에서 fail까지 원문에 있는 경우예요 — 30 contact 포토(Liu 2019 — 3 · 7시 OPEN) · 35 metal 1 뒤
ashing(Chen 2000 — 가운데 gate 누설) · 29 ILD CMP(Goh 2001 — via open, 층을 옮김) · 28 ILD 입자(Stapper 1983 — 층간 short ·
metal 끊김).

*(2026-10-30)* 원문을 다시 읽고 고친 것 —
- **18 세정 헹굼을 뺐어요.** Lin 2012 p.72는 맨 실리콘 위 세정은 절연막이 없어 전하가 거의 안 쌓인다고 해요. 우리 18은
  screen oxide를 벗긴 직후라 그 경우예요.
- **contact 뒤 ashing을 35 metal 1 뒤로 옮겼어요.** Chen 2000은 금속 무늬를 만든 뒤 ashing이고 금속이 전하를 모아요 —
  32 시점엔 아직 금속이 없어요.
- **주 fail 다섯 개를 고쳤어요** — 37 forming gas는 SS만(Mack 2023은 Vt 높음 근거가 없음) · 21b 현상 찌꺼기는 배선 누설만(막이
  남는 쪽) · 08 STI 포토 찌꺼기는 분리 누설 · 01 bevel은 분리 + 배선 누설 반반(판단) · 21a gate 포토 초점은 on 전류(0.5 µm면
  Vt는 수 mV).
- 「원문」 표시가 후한 곳도 적어 뒀어요 — 16 · 19 · 24(Stapper 1983은 일반론) · 12 · 22 · 34(원문 사례는 가장자리가 **덜**
  깎이거나 덜 식각된 방향) · 10 · 26 · 로봇 긁힘(문헌은 모양만, fail은 원리).

## 참고문헌 (복붙용)

```
[1] A. Akheyar, A. Lauwers, R. Lindsay, M. de Potter, G. Tempel, K. Maex, "Analysis of Silicide/Diffusion Contact Resistance Making Use of Transmission Line Structures," Mat. Res. Soc. Symp. Proc., vol. 716, B1.10, 2002. DOI: 10.1557/PROC-716-B1.10
[2] S. Biesemans, S. Kubicek, K. De Meyer, "Test structure to investigate the series resistance components of source/drain structure," IEEE Electron Device Letters, vol. 18, no. 10, pp. 477–479, 1997. DOI: 10.1109/55.624920
[3] A. Y. C. Yu, "Electron tunneling and contact resistance of metal-silicon contact barriers," Solid-State Electronics, vol. 13, pp. 239–247, 1970. DOI: 10.1016/0038-1101(70)90056-0
[4] T. J. Riley, A. K. Nanda, G. Miner, M. F. Pas, S. Hossain-Pas, L. A. Velo, "Evaluation of Applied Materials' rapid thermal processor using SEMATECH methodologies for 0.25 μm technology thermal applications. Part 1," Mat. Res. Soc. Symp. Proc., vol. 429, pp. 15–22, 1996.
[5] L. A. Clevenger, R. W. Mann, R. A. Roy, K. L. Saenger, C. Cabral Jr., J. Piccirillo, "Study of C49-TiSi2 and C54-TiSi2 formation on doped polycrystalline silicon using in situ resistance measurements during annealing," J. Appl. Phys., vol. 76, pp. 7874–7881, 1994. DOI: 10.1063/1.357897
[6] C. Laviron, R. Lindsay, A. Michallet, A. Halimaoui, E. Granneman, "Intra-die temperature non uniformity related to front side emissivity dependance during Rapid Thermal Annealing," ECS 203rd Meeting, Abstract 880, 2003.
[7] MOSIS Parametric Test Results, Run T7CZ (TSMC 0.25 µm, SCN025).
[8] GlobalFoundries GF180MCU open PDK — Electrical Specifications 1.0 (3.3 V NMOS).
[9] S.-T. Lin (Winbond Electronics), "Silicide block for ESD protection devices," US Patent 6,476,449 B1, 2002.
[10] C. H. Stapper, F. M. Armstrong, K. Saji, "Integrated circuit yield statistics," Proceedings of the IEEE, vol. 71, no. 4, pp. 453–470, 1983.
[11] M.-J. Wu, J.-S. R. Jang, J.-L. Chen, "Wafer map failure pattern recognition and similarity ranking for large-scale data sets," IEEE Transactions on Semiconductor Manufacturing, vol. 28, no. 1, pp. 1–12, 2015. DOI: 10.1109/TSM.2014.2364237
```

## 한계

- **칩 측정과 WAT 값은 합성이에요.** WM-811K에는 맵만 있어요. 맵과 CNN 판정만 실측이에요.
- **같은 표 · 같은 식으로 심고 채점했어요.** 그래서 「표와 식이 맞다면 이만큼」의 상한이에요.
- **CNN 오경보** — 문턱 없이 쓰면 처음 보는 lot의 정상 웨이퍼 22,307장 중 1,950장(8.7%)을
  불량 모양으로 불러요(긁힘이 794장). 엔진은 그 웨이퍼에도 원인을 추천해요. 문턱 규칙은 시험만 했어요.
- **fail 근거** — 원인 39개 중 「이 원인이면 칩이 어느 측정으로 떨어지나」를 문헌이 말하는 것은 14개, 일부 1개, 나머지 24개는 소자 원리예요.
- **맵은 WM-811K 양산 맵에서 모양만 빌려 썼어요.** NMOS 공정에서 나온 맵이 아니고, 제품 종류와 직경은 데이터에 없어요.
- **사고가 둘 겹치면 「얼마나」 역산이 틀려요.** 그래서 「다른 측정과 맞나」를 같이 봐요.
- **사고가 겹치면 크기는 말하지 않아요.** 경고만 하고, 후보 단계들의 기록을 같이 보라고 해요.
- **아직 안 한 것** — SAB 단계의 사고, 흩어짐(클린룸) · 긁힘(곡선 vs 직선) · 접합 누설(STI 모서리 — 원문 찾는 중) ·
  정상 웨이퍼 시연, IDDQ에서만 걸린 칩을 뜯어본 원문(Nigh et al., ITC 1998).
