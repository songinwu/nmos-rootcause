"""
N-1. NMOS 지식표 — 「41단계(38 + SAB 3) 중 하나가 잘못되면 바로 짚나」 합성 시험.

원인마다 그 원인의 모양을 가진 실제 WM-811K 맵을 뽑아(원인당 N장), 불량 칩마다 그 원인의 예상 비율대로
측정 결과를 붙이고, 엔진이 그 원인을 되찾는지 본다.

    입력 두 가지 : ① 참 모양(지식표만의 구별력) ② 모양 AI(CNN) 확률(실제 화면과 같은 조건)
    검사 두 가지 : 측정 12칸(rootcause_kb_nmos — 2026-10-28 SAB로 11 → 12) vs 기존 4칸으로 묶은 것
                   (JSON 키 이름의 _11은 처음 11칸일 때 붙인 이름 그대로 둔다)
    채점         : 단독 1위(정답이 1위이고 동점 없음) · 공동 1위 안 · 3순위 안
                   — 「바로 짚음」은 단독 1위다. 표만 보고 한 예측(predicted_unique)과 맞대 본다.

*(2026-10-30 fail 라벨 다시 붙이기)*
- 라벨: 주 fail 50%(둘이면 25 + 25) · 나머지 50%는 다른 칸에 고르게(nk.PRIMARY_SHARE). 칩 10%를 다시 섞던 잡음은 뺐다.
- 맵: **시험용 lot(A-1b lot 단위 분할의 test — CNN이 한 번도 안 본 lot)만** 쓴다. 10-28판은 라벨 맵 전체에서 뽑아
  CNN이 학습한 맵이 섞였다. 시험용 맵이 원인당 장수보다 적은 모양(거의 전체 19장 · 도넛 71장 · 흩어짐 114장 · 긁힘 181장)은
  90° 회전 4방향 × 좌우 뒤집기 8가지로 늘린 뒤 그중에서 뽑는다(라벨은 매번 새로). CNN도 학습 때 같은 회전 · 뒤집기를 썼다.
- 흔들기: 라벨을 주 fail 30 · 70%로 붙이고 엔진은 50% 표로 채점해, 몫이 틀려도 버티는지 본다.

⚠️ 50% 라벨 · 50% 엔진은 심을 때와 채점할 때 같은 표다 → 「표가 맞다면 몇 개를 바로 짚나」의 상한이다(S-1과 같은 한계).

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
from scripts.a1b_lotwise_model import split_by_lot            # noqa: E402
from scripts.s1_synthetic_rootcause import pattern_probs      # noqa: E402
from src.inference import rootcause_kb as kb                  # noqa: E402
from src.inference import rootcause_kb_nmos as nk             # noqa: E402

SEED, N_PER = 42, 200
SHARES = (0.3, 0.5, 0.7)          # 라벨의 주 fail 몫 — 엔진은 늘 nk.PRIMARY_SHARE(0.5) 표로 채점
OUT = "outputs/n1_nmos_identify.json"


def dihedral(m):
    """맵 한 장 → 8가지(90° 회전 4방향 × 좌우 뒤집기)."""
    return [np.rot90(f, k) for f in (m, m[:, ::-1]) for k in range(4)]


def test_pool(maps_u8, y, m_te, pi):
    """시험용 lot에서 모양이 pi인 맵. 원인당 장수보다 적으면 8배로 늘린다."""
    idx = np.where(m_te & (y == pi))[0]
    if len(idx) >= N_PER:
        return maps_u8[idx], len(idx), False
    return np.array([a for i in idx for a in dihedral(maps_u8[i])]), len(idx), True


def plant(maps, sig, rng):
    """불량 칩마다 sig 비율대로 측정 칸을 뽑아 칸별 개수를 센다."""
    counts = np.zeros((len(maps), nk.B))
    for r, m in enumerate(maps):
        n = int((m == 2).sum())
        if n:
            counts[r] = np.bincount(rng.choice(nk.B, size=n, p=sig), minlength=nk.B)
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


def summarize(cs, key):
    solo = np.array([v[key]["solo"] for v in cs])
    return {"identified_ge90": int((solo >= 0.9).sum()), "identified_ge50": int((solo >= 0.5).sum()),
            "mean_solo": float(solo.mean()), "mean_tie_in": float(np.mean([v[key]["tie_in"] for v in cs])),
            "mean_top3": float(np.mean([v[key]["top3"] for v in cs]))}


def main():
    os.chdir(ROOT)
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maps_u8, y, lots, _ = load_cache()
    _, _, m_te = split_by_lot(lots)
    pred11, pred4 = nk.predicted_unique(), nk.predicted_unique(group4=True)
    sigs = {s: nk.signatures(s) for s in SHARES}
    res = {"setup": {"n_per_cause": N_PER, "label_shares": SHARES, "engine_share": nk.PRIMARY_SHARE, "K": nk.K,
                     "bins": nk.B, "seed": SEED, "maps": "A-1b lot 단위 분할의 test lot만 (적은 모양은 회전 · 뒤집기 8배)",
                     "note": "50% 라벨 · 50% 엔진은 같은 표 — 상한"}, "pools": {}, "causes": {}}
    pools, probs = {}, {}
    for c in nk.CAUSES:
        if c.pattern not in pools:
            pi = kb.PATTERNS.index(c.pattern)
            pool, n_orig, aug = test_pool(maps_u8, y, m_te, pi)
            pools[c.pattern] = pool
            probs[c.pattern] = pattern_probs(pool.astype(np.float32) / 2.0, device)
            res["pools"][c.pattern] = {"test_maps": int(n_orig), "augmented": bool(aug), "pool": int(len(pool))}
    for ci, c in enumerate(nk.CAUSES):
        pi = kb.PATTERNS.index(c.pattern)
        pool, Pall = pools[c.pattern], probs[c.pattern]
        idx = rng.choice(len(pool), size=N_PER, replace=len(pool) < N_PER)
        maps, Pc = pool[idx], Pall[idx]
        onehot = np.zeros((N_PER, len(kb.PATTERNS)))
        onehot[:, pi] = 1.0
        counts = plant(maps, sigs[nk.PRIMARY_SHARE][ci], rng)
        r = {"name": c.name, "pattern": c.pattern, "evidence": c.evidence, "fail_evidence": nk.fail_evidence(c.id)[0],
             "pred_unique11": pred11[c.id], "pred_unique4": pred4[c.id],
             "true_pattern_11": judge(onehot, counts, ci, None),
             "true_pattern_4": judge(onehot, nk.collapse4(counts), ci, nk.SIGNATURES4),
             "cnn_11": judge(Pc, counts, ci, None),
             "cnn_4": judge(Pc, nk.collapse4(counts), ci, nk.SIGNATURES4),
             # 비교 기준(2026-10-30) — 검사 결과 없이 모양만 넣으면 몇 개를 짚나. 난수를 쓰지 않아 위 숫자는 그대로다
             "pattern_only": judge(onehot, np.zeros_like(counts), ci, None),
             "cnn_pattern_only": judge(Pc, np.zeros_like(counts), ci, None),
             "cnn_pattern_acc": float((Pc.argmax(1) == pi).mean())}
        for s in SHARES:
            if s == nk.PRIMARY_SHARE:
                continue
            cs_ = plant(maps, sigs[s][ci], rng)
            r[f"share{int(s * 100)}_true_pattern_11"] = judge(onehot, cs_, ci, None)
            r[f"share{int(s * 100)}_cnn_11"] = judge(Pc, cs_, ci, None)
        res["causes"][c.id] = r
        print(f"{c.id:24s} {c.pattern:9s} 예측 {'단독' if pred11[c.id] else '묶음'} | 참모양 {nk.B}칸 단독 "
              f"{r['true_pattern_11']['solo']*100:5.1f}% · 4칸 {r['true_pattern_4']['solo']*100:5.1f}% | CNN {nk.B}칸 단독 "
              f"{r['cnn_11']['solo']*100:5.1f}% 공동 안 {r['cnn_11']['tie_in']*100:5.1f}% | 30% {r['share30_cnn_11']['solo']*100:5.1f}"
              f" · 70% {r['share70_cnn_11']['solo']*100:5.1f}")
    cs = list(res["causes"].values())
    summ = {k: summarize(cs, k) for k in ("true_pattern_11", "true_pattern_4", "cnn_11", "cnn_4",
                                          "pattern_only", "cnn_pattern_only")}
    # 무작위로 하나 고르면: 단독 1위 1/K · 3순위 안 3/K — 원인마다 같으므로 90% 넘는 원인은 0개
    summ["random"] = {"identified_ge90": 0, "identified_ge50": 0, "mean_solo": 1 / nk.K, "mean_tie_in": 1 / nk.K,
                      "mean_top3": 3 / nk.K}
    for s in SHARES:
        if s != nk.PRIMARY_SHARE:
            for k in ("true_pattern_11", "cnn_11"):
                summ[f"share{int(s * 100)}_{k}"] = summarize(cs, f"share{int(s * 100)}_{k}")
    agree = sum(1 for v in cs if v["pred_unique11"] == (v["true_pattern_11"]["solo"] >= 0.9))
    summ["prediction_agreement_11"] = f"{agree}/{nk.K}"
    summ["cnn_pattern_acc_by_pattern"] = {p: float(np.mean([v["cnn_pattern_acc"] for v in cs if v["pattern"] == p]))
                                          for p in pools}
    res["summary"] = summ
    print(json.dumps({k: v for k, v in summ.items()}, ensure_ascii=False, indent=1))
    with open(OUT, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("저장:", OUT)


if __name__ == "__main__":
    main()
