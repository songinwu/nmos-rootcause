# NMOS 원인 공정 추천 — 지식표 · 「얼마나」 · 합성 시험 · 시연 화면

종합설계 「웨이퍼 불량 지도로 원인 공정 좁히기」에서 **NMOS 부분만** 뽑아 온 저장소예요.
엔진 본체 · 대시보드는 비공개 저장소에 있어서, 원인 표 쪽 파일은 **읽기용**이고 혼자서는 실행되지 않아요.
**모양 CNN 학습 코드는 여기서 그대로 돌아가요** — 아래 [모양 CNN](#모양-cnn--학습-코드-2026-10-30)을 보세요.

> **최신 — 2026-10-30** · 모양 CNN 학습 코드 추가.
> **2026-10-28** · 41단계(38 + SAB 3) · 측정 12칸 · 「얼마나」를 말하는 원인 8개 + 결함 밀도 · 시연 웨이퍼 17장.
> `docs/`의 PDF는 2판(38단계 · SAB 넣기 전)이에요. 그 뒤에 바뀐 점은 아래 [SAB 추가](#sab-추가-2026-10-28)를 보세요.

## 무엇을 했나

1. **공정을 정했어요.** NMOS 하나를 41단계 · 마스크 6장으로 만들어요. IDEC 0.5 µm 공정에서 NMOS 부분만 가져오고,
   소자 분리는 STI(KSIA)로, LDD는 빼고, silicide를 막는 층(SAB)을 넣었어요.
2. **원인 표를 만들었어요.** 단계마다 「틀어지면 **웨이퍼맵이 어떤 모양**이 되고 **칩 측정 중 무엇이 바뀌나**」를 원인 40개로 적었어요.
3. **「어느 단계」** — 엔진은 맵 모양으로 후보를 좁히고, 칩 측정 12칸으로 그중에서 골라요.
4. **「얼마나」** — 웨이퍼 WAT 값(Vt · SS · on 전류 · C-V 산화막 · 면저항 셋)을 MOS 식으로 거꾸로 풀어
   「gate 산화막 약 75 Å, 정상 53 ~ 68 Å 밖」처럼 파라미터가 얼마나 벗어났는지 말해요.

**EDS 맵이 「어느 단계」를, WAT 값이 「얼마나」를 맡아요.** 장비 설정(산화로 몇 °C · 몇 분)은 같은 결과를 만드는 조합이
여러 개라 정하지 않고 장비 기록으로 넘겨요. 예외는 RTA예요 — 면저항과 온도의 관계를 공장이 실제로 이렇게 써요(Riley 1996).

## 먼저 읽을 것

1. 이 README
2. [`src/inference/rootcause_kb_nmos.py`](src/inference/rootcause_kb_nmos.py) — 41단계(`STEPS`)와 원인 40개(`CAUSES`)
3. [`src/inference/nmos_device.py`](src/inference/nmos_device.py) — 「얼마나」 식 · 기준값 · 합격선 · 불량 경계 (맨 위 설명에 출처)
4. [`docs/NMOS_원인공정_지식표_조원설명.pdf`](docs/NMOS_원인공정_지식표_조원설명.pdf) — 단계 · 측정 · 연결표 · 근거 (2판, 38단계 기준)

## 파일

| 파일 | 내용 |
|---|---|
| `src/inference/rootcause_kb_nmos.py` | 지식표 — 41단계, 측정 12칸, 원인 40개(단계 · 무엇이 틀어지나 · 바뀌는 측정 · 맵 모양 · 근거) |
| `src/inference/nmos_device.py` | 「얼마나」 — 공정값 ↔ WAT 값 식, 기준값 · 합격선 · 불량 경계, 역산(중앙값 · 90% 범위 · 다른 측정과 맞나) |
| `scripts/n1_nmos_identify.py` | 합성 시험 N-1 — 원인마다 실제 맵 200장에 사고를 심고 엔진이 되찾는지 봄 |
| `scripts/make_nmos_scenarios.py` | 시연 웨이퍼 17장(N1 ~ N17) — 맵 · CNN 판정 · 칩 측정 · WAT 값 · 크기 · 경고 · 화면 문구 |
| `data/demo/nmos_scenarios.json` | 위 스크립트가 만든 결과 |
| `src/dashboard/demo16.py` | 대시보드 섹션 16 시연 화면 (MOSCAP · CMOS · NMOS) |
| `tests/test_rootcause_nmos.py` · `tests/test_nmos_device.py` · `tests/test_demo16_text.py` | 표 모양 · 대표 판정 · 크기 역산 · 시나리오가 엔진과 맞는지 |
| `docs/…조원설명.pdf` | 조원 설명용 정리본 (2판) |
| `src/training/train_wafermap.py` | 모양 CNN — 모델 구조 · WM-811K 읽기(26×26) · 웨이퍼 단위 분할 학습 |
| `scripts/a1b_lotwise_model.py` · `scripts/a1_lot_consistency.py` | 모양 CNN — **lot 단위 분할 학습(시연 · 엔진이 쓰는 것)** · lot 뭉침 검정 |
| `src/inference/wafermap_predictor.py` | 학습한 CNN으로 맵 한 장 판정 |
| `scripts/download_wm811k.sh` · `requirements-cnn.txt` | WM-811K 받기(Kaggle) · 필요한 패키지 |
| `results/cnn_lotwise.json` · `results/cnn_waferwise.json` | 두 학습의 결과 숫자 |

엔진 본체(`rootcause_kb.score_batch`)는 이 저장소에 없어요.
엔진이 하는 일은 「모양으로 고른 후보 중에서, 칩 측정이 그 원인의 예상과 가장 잘 맞는 순서로 줄 세우기」예요.

## 결과 1 — 「어느 단계」를 바로 짚나 (합성 시험 N-1)

원인마다 그 모양의 실제 WM-811K 맵 200장에 사고를 심었어요(칩의 10%는 아무 칸 — 잡음).
엔진이 그 원인을 **단독 1위**(동점 없이 1위)로 되찾는 비율이 90% 이상이면 「바로 짚음」으로 셌어요.

| 조건 | 측정 12칸 | 기존 4칸 |
|---|---|---|
| 맵 모양을 정답으로 줬을 때 | **23개** | 10개 |
| 모양 CNN을 거쳤을 때 | **23개** | 9개 |

- 나머지 17개는 2 ~ 4개 묶음으로 남아요. 정답이 공동 1위 안에 들어 있는 비율은 99.96%(모양 CNN을 거치면 97.5%)예요.
- 묶음은 모양도 측정도 원리상 같아서 못 가르는 거예요. 공정 중 측정이나 장비 기록으로 갈라요(PDF 4절).
- 표만 보고 「단독일 것」이라고 한 예측과 40개 모두 일치했어요.

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
| 장비 변수 | 7 | ❌ 칩 측정에 안 남음 → 장비 기록 |

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
| N2 | 거의 전체 + SS 큼 · Vt 높음 | 37 forming gas | 계면 트랩 ×116 → ×119 |
| N3 | 거의 전체 + poly 면저항 · Vt | 20 poly 도핑 | 5.0×10¹⁸ → 5.1×10¹⁸ cm⁻³ |
| N4 | 가장자리 링 + S/D 면저항 | 26 RTA | −22 °C → −20 °C ※ |
| N5 | 가장자리 일부 + 연결 끊김 | 30 contact 포토 | — (여유) |
| N6 | 도넛 + 배선 누설 · 끊김 | 21 gate 포토 현상 | — (장비 변수) |
| N7 | 가장자리 링 + 연결 끊김 | 29 · 31 공동 | — (ILD 두께 vs ring RF 시간으로 가름) |
| N8 | 가운데 + gate 누설 | 18 · 23 · 32 공동 | — (헹굼물 저항 vs ashing 장비) |
| N9 | 거의 전체 + Vt · on 전류 | 17 · 19 공동 | 산화막 75 Å → 75 Å (둘 다 같은 크기) |
| N10 | 거의 전체 + S/D 면저항 | 24 S/D 주입 | 도즈 −35% → −36% |
| N11 | 거의 전체 + silicide 면저항 | 27 salicide | 남은 C49 35% → 35% |
| N12 | 한 곳 뭉침 + 분리 누설 | 10 trench 채우기 | 그 자리 칩당 결함 약 0.8개 (나머지 0.13개) — 실제 맵에서 |
| N13 | 거의 전체 + Vt · SS (사고 둘) | 37 forming gas · 2순위 16 Vt 주입 | ⚠️ 「파라미터 하나로는 설명이 안 됨」 경고 |
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
  15,000장으로 깎고, 클래스 가중 loss · 회전 · 반전을 써요. **데이터와 학습한 가중치는 저장소에 없어요** — 데이터는
  `scripts/download_wm811k.sh`로 Kaggle에서 받아요.

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
| 원인 → 맵 모양 | 원인 40개 중 문헌 23개(원문 16 · 일부 3 · 옮김 4), 강의 5 · 원리 12 |
| 원인 → 바뀌는 측정 | 대부분 원리 추론. 면저항 쪽은 원문 5편으로 다시 봄(아래) |
| 측정이 얼마나 쏠리나 | 근거 없음 — 주 측정 약 55%는 우리가 정한 값 |
| 기준값 · 합격선 | 공개 실측 · 공개 spec 폭 + 일부 가정 (위 표) |

가장 단단한 것은 같은 단계에서 측정까지 원문에 있는 경우예요 — 18 세정(Lin 2012, TSMC) · 30 contact 포토(Liu 2019) ·
29 ILD CMP(Goh 2001, 층을 옮김). 「원문」 표시가 후하게 붙은 것도 있어요 — 16 · 19 · 24(Stapper 1983은 「공정 시간 · 온도가
틀어지면 웨이퍼 전체가 떨어진다」는 일반론) · 26(가장자리 링 모양은 원문, 면저항 쪽은 Riley 1996으로 보강) · 31(via · DRAM 구멍
사례를 contact로 옮김) · 37(MOS 커패시터 실험).

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
- **맵은 WM-811K 양산 맵에서 모양만 빌려 썼어요.** NMOS 공정에서 나온 맵이 아니고, 제품 종류와 직경은 데이터에 없어요.
- **사고가 둘 겹치면 「얼마나」 역산이 틀려요.** 그래서 「다른 측정과 맞나」를 같이 봐요.
- **사고가 겹치면 크기는 말하지 않아요.** 경고만 하고, 후보 단계들의 기록을 같이 보라고 해요.
- **아직 안 한 것** — 21a gate 포토 초점의 측정 방향(Vt → on 전류), SAB 단계의 사고, 흩어짐(클린룸) · 긁힘(곡선 vs 직선) ·
  접합 누설(STI 모서리 — 원문 찾는 중) · 정상 웨이퍼 시연, PDF 3판.
