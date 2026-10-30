"""
A-1d. 패턴 CNN 오경보 줄이기 — 모양별 「불량 판정 문턱」을 검증용 lot에서 정하고 시험용 lot에서 잰다.

A-1c(lot 단위 시험)에서 recall만 보면 긁힘 72% · 한 곳 뭉침 71%지만, **precision은 긁힘 13% · 한 곳 뭉침 51% ·
가장자리 일부 52%**였다 — 정상 웨이퍼 22,307장 중 794장을 긁힘으로 불렀다. 엔진은 CNN 1위가 none이 아니면
원인을 추천하므로(rootcause_kb.py `top_p == "none"`), 이 오경보는 그대로 「정상 웨이퍼에 원인 공정 추천」이 된다.

규칙 : CNN 1위가 불량 모양 c인데 그 확률이 문턱 τ_c보다 낮으면 none으로 본다(정상으로 넘김).
문턱 : **검증용 lot(15%)** 에서 모양마다 F1이 가장 높은 τ_c를 고른다 — 시험용 lot은 고를 때 쓰지 않는다.
보고 : 시험용 lot(15%)에서 문턱 전후의 precision · recall(lot 부트스트랩 95% 구간), 정상 → 불량 오경보 수,
       불량 → 정상으로 놓친 수.

    conda run -n semi_ai python -m scripts.a1d_cnn_threshold        # 1분 안 (GPU)
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.a1_lot_consistency import load_cache                     # noqa: E402
from scripts.a1b_lotwise_model import split_by_lot                     # noqa: E402
from scripts.s1_synthetic_rootcause import pattern_probs               # noqa: E402
from src.training.train_wafermap import CLASS_NAMES                    # noqa: E402

OUT = "outputs/a1d_cnn_threshold.json"
N_BOOT, SEED = 1000, 42
GRID = [0.0] + [round(x, 2) for x in np.arange(0.30, 1.00, 0.01)]     # 0.0 = 문턱 없음(argmax 그대로)


def gated(P, tau):
    """argmax 뒤, 1위 확률이 그 모양의 문턱보다 낮으면 none(0)으로."""
    yp = P.argmax(1)
    low = P[np.arange(len(P)), yp] < tau[yp]
    return np.where((yp != 0) & low, 0, yp)


def pick_thresholds(P, y):
    K = P.shape[1]
    tau = np.zeros(K)
    yp0 = P.argmax(1)
    for c in range(1, K):
        best = (-1.0, 0.0)
        for t in GRID:
            yp = np.where((yp0 == c) & (P[:, c] < t), 0, yp0)
            f = f1_score(y == c, yp == c, zero_division=0)
            if f > best[0] + 1e-9:
                best = (f, t)
        tau[c] = best[1]
    return tau


def metrics(y, yp, lots, K):
    prec, rec, _, sup = precision_recall_fscore_support(y, yp, labels=range(K), zero_division=0)
    uniq = np.unique(lots)
    idx_of = {l_: np.where(lots == l_)[0] for l_ in uniq}
    rng = np.random.default_rng(SEED)
    bp, br = [], []
    for _ in range(N_BOOT):
        ii = np.concatenate([idx_of[l_] for l_ in rng.choice(uniq, size=len(uniq), replace=True)])
        p_, r_, _, _ = precision_recall_fscore_support(y[ii], yp[ii], labels=range(K), zero_division=0)
        bp.append(p_)
        br.append(r_)
    pci = np.percentile(np.array(bp), [2.5, 97.5], axis=0)
    rci = np.percentile(np.array(br), [2.5, 97.5], axis=0)
    per = {CLASS_NAMES[k]: dict(n=int(sup[k]), precision=round(float(prec[k]), 4), recall=round(float(rec[k]), 4),
                                precision_ci95=[round(float(pci[0][k]), 4), round(float(pci[1][k]), 4)],
                                recall_ci95=[round(float(rci[0][k]), 4), round(float(rci[1][k]), 4)])
           for k in range(K)}
    return dict(accuracy=round(float((y == yp).mean()), 4),
                macro_f1=round(float(f1_score(y, yp, average="macro", labels=range(K))), 4),
                normal_flagged=int(((y == 0) & (yp != 0)).sum()),          # 정상인데 불량 모양으로 → 원인 추천이 나감
                defect_missed=int(((y != 0) & (yp == 0)).sum()),           # 불량인데 정상으로 → 원인 추천이 안 나감
                n_normal=int((y == 0).sum()), n_defect=int((y != 0).sum()), per_class=per)


def main():
    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maps_u8, y, lots, _ = load_cache()
    _, m_va, m_te = split_by_lot(lots)
    va, te = np.where(m_va)[0], np.where(m_te)[0]
    P_va = pattern_probs(maps_u8[va].astype(np.float32) / 2.0, device)
    P_te = pattern_probs(maps_u8[te].astype(np.float32) / 2.0, device)
    K = len(CLASS_NAMES)

    tau = pick_thresholds(P_va, y[va])
    before = metrics(y[te], P_te.argmax(1), lots[te], K)
    after = metrics(y[te], gated(P_te, tau), lots[te], K)
    res = {"setup": {"pick_on": f"validation lots {len(np.unique(lots[va])):,} · wafers {len(va):,}",
                     "report_on": f"test lots {len(np.unique(lots[te])):,} · wafers {len(te):,}",
                     "rule": "1위가 불량 모양 c이고 P(c) < τ_c 이면 none", "criterion": "검증용 lot에서 모양별 F1 최대",
                     "bootstrap": {"unit": "test lot", "n": N_BOOT, "seed": SEED}},
           "thresholds": {CLASS_NAMES[k]: float(tau[k]) for k in range(1, K)},
           "before": before, "after": after}

    print(f"문턱(검증용 lot에서 고름): " + " · ".join(f"{CLASS_NAMES[k]} {tau[k]:.2f}" for k in range(1, K)))
    print(f"정상 → 불량 오경보 {before['normal_flagged']:,} → {after['normal_flagged']:,}장 (정상 {before['n_normal']:,}장 중) · "
          f"불량 → 정상 놓침 {before['defect_missed']:,} → {after['defect_missed']:,}장 (불량 {before['n_defect']:,}장 중)")
    print(f"정확도 {before['accuracy']:.3f} → {after['accuracy']:.3f} · macro F1 {before['macro_f1']:.3f} → {after['macro_f1']:.3f}")
    print(f"{'모양':<10}{'n':>6}  precision 전 → 후            recall 전 → 후")
    for k in range(1, K):
        b, a = before["per_class"][CLASS_NAMES[k]], after["per_class"][CLASS_NAMES[k]]
        print(f"{CLASS_NAMES[k]:<10}{b['n']:>6}  {b['precision']:.3f} → {a['precision']:.3f} {a['precision_ci95']}   "
              f"{b['recall']:.3f} → {a['recall']:.3f} {a['recall_ci95']}")
    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("저장:", OUT)


if __name__ == "__main__":
    main()
