"""
N-1. NMOS 지식표 — 「41단계(38 + SAB 3) 중 하나가 잘못되면 바로 짚나」 합성 시험.

원인 40개마다 그 원인의 모양을 가진 실제 WM-811K 맵을 뽑아(원인당 N장), 불량 칩마다 그 원인의 예상 비율대로
측정 결과를 붙이고(칩의 10%는 아무 결과 — 잡음), 엔진이 그 원인을 되찾는지 본다.

    입력 두 가지 : ① 참 모양(지식표만의 구별력) ② 모양 AI(CNN) 확률(실제 화면과 같은 조건)
    검사 두 가지 : 측정 12칸(rootcause_kb_nmos — 2026-10-28 SAB로 11 → 12) vs 기존 4칸으로 묶은 것
                   (JSON 키 이름의 _11은 처음 11칸일 때 붙인 이름 그대로 둔다)
    채점         : 단독 1위(정답이 1위이고 동점 없음) · 공동 1위 안 · 3순위 안
                   — 「바로 짚음」은 단독 1위다. 표만 보고 한 예측(predicted_unique)과 맞대 본다.

⚠️ 심을 때와 채점할 때 같은 표를 쓴다 → 「표가 맞다면 몇 개를 바로 짚나」의 상한이다(S-1과 같은 한계).

    conda run -n semi_ai python -m scripts.n1_nmos_identify
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.a1_lot_consistency import load_cache            # noqa: E402
from scripts.s1_synthetic_rootcause import pattern_probs      # noqa: E402
from src.inference import rootcause_kb as kb                  # noqa: E402
from src.inference import rootcause_kb_nmos as nk             # noqa: E402

SEED, N_PER, EPS_BIN = 42, 200, 0.10
OUT = "outputs/n1_nmos_identify.json"


def plant(maps_u8, idx, cause_i, rng):
    counts = np.zeros((len(idx), nk.B))
    for r, i in enumerate(idx):
        n = int((maps_u8[i] == 2).sum())
        if n == 0:
            continue
        b = rng.choice(nk.B, size=n, p=nk.SIGNATURES[cause_i])
        flip = rng.random(n) < EPS_BIN
        b[flip] = rng.integers(0, nk.B, int(flip.sum()))
        counts[r] = np.bincount(b, minlength=nk.B)
    return counts


def judge(P, counts, truth, signatures):
    sc = nk.score(P, counts, signatures=signatures)
    top_pat = 1 + P[:, 1:].argmax(1)
    sc = np.where(nk.eligible(kb.softmax(sc), top_pat), sc, -np.inf)
    post = kb.softmax(sc)
    order = np.argsort(-sc, axis=1, kind="stable")
    rank = (order == truth).argmax(1)
    rank = np.where(np.isfinite(sc[:, truth]), rank, nk.K)
    tie = post >= post.max(1, keepdims=True) * (1 - kb.TIE_RTOL)
    ntie = tie.sum(1)
    solo = (rank == 0) & (ntie == 1)
    in_tie = tie[:, truth]
    return {"solo": float(solo.mean()), "tie_in": float(in_tie.mean()), "top3": float((rank < 3).mean()),
            "mean_tie_size": float(ntie.mean())}


def main():
    os.chdir(ROOT)
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maps_u8, y, lots, _ = load_cache()
    pred11, pred4 = nk.predicted_unique(), nk.predicted_unique(group4=True)
    res = {"setup": {"n_per_cause": N_PER, "eps_bin": EPS_BIN, "K": nk.K, "bins": nk.B, "seed": SEED,
                     "note": "심을 때와 채점할 때 같은 표 — 상한"}, "causes": {}}
    for ci, c in enumerate(nk.CAUSES):
        pi = kb.PATTERNS.index(c.pattern)
        pool = np.where(y == pi)[0]
        idx = rng.choice(pool, size=N_PER, replace=len(pool) < N_PER)
        counts = plant(maps_u8, idx, ci, rng)
        onehot = np.zeros((N_PER, len(kb.PATTERNS)))
        onehot[:, pi] = 1.0
        Pc = pattern_probs(maps_u8[idx].astype(np.float32) / 2.0, device)
        r = {"name": c.name, "pattern": c.pattern, "evidence": c.evidence,
             "pred_unique11": pred11[c.id], "pred_unique4": pred4[c.id],
             "true_pattern_11": judge(onehot, counts, ci, None),
             "true_pattern_4": judge(onehot, nk.collapse4(counts), ci, nk.SIGNATURES4),
             "cnn_11": judge(Pc, counts, ci, None),
             "cnn_4": judge(Pc, nk.collapse4(counts), ci, nk.SIGNATURES4),
             "cnn_pattern_acc": float((Pc.argmax(1) == pi).mean())}
        res["causes"][c.id] = r
        print(f"{c.id:24s} {c.pattern:9s} 예측 {'단독' if pred11[c.id] else '묶음'} | 참모양 {nk.B}칸 단독 {r['true_pattern_11']['solo']*100:5.1f}% "
              f"· 4칸 {r['true_pattern_4']['solo']*100:5.1f}% | CNN {nk.B}칸 단독 {r['cnn_11']['solo']*100:5.1f}% 3순위 {r['cnn_11']['top3']*100:5.1f}%")
    cs = res["causes"].values()
    summ = {}
    for key in ("true_pattern_11", "true_pattern_4", "cnn_11", "cnn_4"):
        solo = np.array([v[key]["solo"] for v in cs])
        summ[key] = {"identified_ge90": int((solo >= 0.9).sum()), "identified_ge50": int((solo >= 0.5).sum()),
                     "mean_solo": float(solo.mean()), "mean_tie_in": float(np.mean([v[key]["tie_in"] for v in cs])),
                     "mean_top3": float(np.mean([v[key]["top3"] for v in cs]))}
    agree = sum(1 for v in cs if v["pred_unique11"] == (v["true_pattern_11"]["solo"] >= 0.9))
    summ["prediction_agreement_11"] = f"{agree}/{nk.K}"
    res["summary"] = summ
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("저장:", OUT)


if __name__ == "__main__":
    main()
