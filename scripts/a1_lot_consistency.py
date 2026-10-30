"""
A-1. Lot 일관성 검정 — wafer defect map 원인 공정 추천의 1단계 검증.

가설: 같은 원인으로 생긴 불량은 같은 lot에 몰린다.
      → lot을 원인의 대리 정답(proxy ground truth)으로 쓸 수 있다.

A-1a (이 파일): **모델 없이 데이터만** 본다. WM-811K의 실제 failureType 라벨이
      lot 단위로 뭉치는지 permutation test로 검정한다. 학습/검증 누수가 원천적으로 없다.
A-1b (a1b_lotwise_model.py): lot 단위로 쪼갠 split으로 CNN을 재학습해,
      **처음 보는 lot**에서도 예측이 뭉치는지 본다.

검정: 라벨을 전역 셔플해 귀무분포를 만든다. lot 크기 분포와 라벨 주변분포가 보존되므로
      'none이 85%라 purity가 원래 높다'는 교란이 귀무분포에 그대로 들어간다.
"""
import os, json
import numpy as np

MAP_SIZE = 26
N_CLASSES = 9
CLASS_NAMES = ["none", "center", "donut", "edge-loc", "edge-ring",
               "loc", "near-full", "random", "scratch"]
PKL = "data/raw/wm811k/WM811K.pkl"
CACHE = "data/processed/wm811k_labeled_lot26.npz"


def build_cache():
    import pandas as pd
    from PIL import Image

    print(f"pkl 로드: {PKL} (수 분 소요)")
    df = pd.read_pickle(PKL)
    print(f"  전체 {len(df)} rows")
    label_map = {n: i for i, n in enumerate(CLASS_NAMES)}

    def norm_label(ft):
        while isinstance(ft, (list, np.ndarray)):
            if len(ft) == 0:
                return None
            ft = ft[0]
        s = str(ft).strip().lower()
        return s if s in label_map else None

    labels_raw = df["failureType"].map(norm_label)
    mask = labels_raw.notna()
    df_lab = df[mask]
    labels_str = labels_raw[mask]
    idx = df_lab.index.to_numpy()
    print(f"  라벨 있는 맵: {len(idx)}")

    maps = np.zeros((len(idx), MAP_SIZE, MAP_SIZE), dtype=np.uint8)
    y = np.zeros(len(idx), dtype=np.int64)
    lots = []
    for n, i in enumerate(idx):
        wm = np.asarray(df_lab.at[i, "waferMap"]).astype(np.uint8)
        img = Image.fromarray(wm, mode="L").resize((MAP_SIZE, MAP_SIZE), Image.NEAREST)
        maps[n] = np.array(img, dtype=np.uint8)
        y[n] = label_map[labels_str.at[i]]
        lots.append(str(df_lab.at[i, "lotName"]))
        if (n + 1) % 25000 == 0:
            print(f"  resize {n+1}/{len(idx)}")

    lot_names, lot_codes = np.unique(np.array(lots), return_inverse=True)
    os.makedirs("data/processed", exist_ok=True)
    np.savez_compressed(CACHE, maps=maps, y=y, lot_codes=lot_codes, lot_names=lot_names)
    print(f"  캐시 저장: {CACHE}  (lot {len(lot_names)}개)")
    return maps, y, lot_codes, lot_names


def load_cache():
    if os.path.exists(CACHE):
        print(f"캐시 로드: {CACHE}")
        z = np.load(CACHE, allow_pickle=True)
        return z["maps"], z["y"], z["lot_codes"], z["lot_names"]
    return build_cache()


# ──────────────────────────────────────────────
# 통계
# ──────────────────────────────────────────────

def lot_counts(lot_idx, y, n_lots):
    flat = np.bincount(lot_idx * N_CLASSES + y, minlength=n_lots * N_CLASSES)
    return flat.reshape(n_lots, N_CLASSES)


def purity_entropy(counts):
    n = counts.sum(axis=1)
    purity = counts.max(axis=1) / n
    p = counts / n[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.where(p > 0, np.log(p), 0.0)
    ent = -(p * lp).sum(axis=1) / np.log(N_CLASSES)
    return purity, ent, n


def permutation_test(tag, lot_raw, y, n_perm=2000, seed=0, min_size=2):
    """lot 크기 >= min_size인 lot만 대상. 라벨 전역 셔플로 귀무분포 생성."""
    _, lot_idx = np.unique(lot_raw, return_inverse=True)
    sizes = np.bincount(lot_idx)
    keep = sizes[lot_idx] >= min_size
    lot_idx, y = lot_idx[keep], y[keep]
    _, lot_idx = np.unique(lot_idx, return_inverse=True)
    n_lots = lot_idx.max() + 1 if len(lot_idx) else 0
    if n_lots == 0:
        print(f"\n### {tag}\n  (대상 없음)")
        return None

    pu, en, n = purity_entropy(lot_counts(lot_idx, y, n_lots))
    obs_p = float((pu * n).sum() / n.sum())
    obs_e = float((en * n).sum() / n.sum())

    rng = np.random.default_rng(seed)
    null_p = np.empty(n_perm); null_e = np.empty(n_perm)
    for b in range(n_perm):
        c = lot_counts(lot_idx, rng.permutation(y), n_lots)
        p2, e2, n2 = purity_entropy(c)
        null_p[b] = (p2 * n2).sum() / n2.sum()
        null_e[b] = (e2 * n2).sum() / n2.sum()

    pval = (np.sum(null_p >= obs_p) + 1) / (n_perm + 1)
    sd = null_p.std(ddof=1)
    z = (obs_p - null_p.mean()) / sd if sd > 0 else float("inf")
    # 정규화 효과크기: 귀무 대비 얼마나 상한(1.0)에 다가갔나
    gain = (obs_p - null_p.mean()) / (1.0 - null_p.mean())

    res = dict(tag=tag, n_lots=int(n_lots), n_wafers=int(len(y)),
               obs_purity=obs_p, null_purity_mean=float(null_p.mean()),
               null_purity_sd=float(sd), p_value=float(pval), z=float(z),
               gain=float(gain), obs_entropy=obs_e,
               null_entropy_mean=float(null_e.mean()), n_perm=n_perm)
    print(f"\n### {tag}")
    print(f"  lot {res['n_lots']:,} / wafer {res['n_wafers']:,}")
    print(f"  관측 purity {obs_p:.4f}  |  귀무 {null_p.mean():.4f} ± {sd:.4f}  "
          f"|  z = {z:,.0f}  p = {pval:.2e}")
    print(f"  엔트로피 {obs_e:.4f} (귀무 {null_e.mean():.4f}, {100*(1-obs_e/null_e.mean()):.1f}% 감소)")
    print(f"  귀무→상한 거리의 {gain*100:.1f}%를 메움")
    return res


def lot_lift(lot_raw, y):
    """패턴별: 한 장이 X면 같은 lot의 나머지도 X일 확률 vs 전체에서 X의 비율."""
    _, lot_idx = np.unique(lot_raw, return_inverse=True)
    n_lots = lot_idx.max() + 1
    counts = lot_counts(lot_idx, y, n_lots)
    n = counts.sum(axis=1)
    rows, total = [], len(y)
    for k in range(N_CLASSES):
        sel = y == k
        m = int(sel.sum())
        if m == 0:
            continue
        others_k = counts[lot_idx[sel], k] - 1
        others_n = n[lot_idx[sel]] - 1
        ok = others_n > 0
        cond = float((others_k[ok] / others_n[ok]).mean()) if ok.sum() else float("nan")
        base = m / total
        rows.append(dict(cls=CLASS_NAMES[k], n=m, base=base, cond=cond,
                         lift=cond / base, n_used=int(ok.sum())))
    return rows


def print_lift(rows, title):
    print(f"\n### {title}")
    print(f"  {'pattern':<11}{'n':>8}{'전체비율':>10}{'같은lot':>10}{'lift':>9}")
    for r in rows:
        print(f"  {r['cls']:<11}{r['n']:>8,}{r['base']*100:>9.2f}%"
              f"{r['cond']*100:>9.2f}%{r['lift']:>8.1f}x")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    maps, y, lot_codes, lot_names = load_cache()
    print(f"라벨 웨이퍼 {len(y):,}장 / lot {len(lot_names):,}개")
    print("  실제 분포:", {CLASS_NAMES[i]: int((y == i).sum()) for i in range(N_CLASSES)})
    sizes = np.bincount(np.unique(lot_codes, return_inverse=True)[1])
    print(f"  lot 크기: 중앙값 {np.median(sizes):.0f} / 평균 {sizes.mean():.1f} / 최대 {sizes.max()}")

    out = {"n_wafers": int(len(y)), "n_lots": int(len(lot_names))}
    tests = []
    tests.append(permutation_test("A-1a-① 실제 라벨 — 전체 (9종)", lot_codes, y))
    nn = y != 0
    tests.append(permutation_test("A-1a-② 실제 라벨 — none 제외 (결함 8종)", lot_codes[nn], y[nn]))
    # 결함 유무만 (이진) — '어떤 패턴인가' 이전에 '불량이 뜨는가'가 뭉치는지
    yb = (y != 0).astype(np.int64)
    tests.append(permutation_test("A-1a-③ 결함 유무만 (2종)", lot_codes, yb))
    out["tests"] = [t for t in tests if t]

    rows = lot_lift(lot_codes, y)
    print_lift(rows, "A-1a-④ 패턴별 lot 집중도 — 한 장이 X면 같은 lot 나머지도 X일 확률")
    out["lift"] = rows

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/a1a_lot_consistency.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n저장: outputs/a1a_lot_consistency.json")
