"""
A-1b. **lot 단위 split**으로 CNN을 재학습하고, 처음 보는 lot에서 예측이 뭉치는지 본다.

왜 재학습하나 — 기존 체크포인트(models/wafermap/wafermap_cnn.pt)는 라벨 있는 결함
웨이퍼 25,519장을 **전부** 학습에 썼다. 그래서 held-out에 결함이 한 장도 남지 않고,
'예측이 lot 단위로 뭉친다'를 이미 본 lot에서 확인하는 꼴이 된다.
여기서는 **lot을 통째로** train/val/test로 갈라 test lot을 학습에서 완전히 뺀다.

원본과 같게 유지한 것: 아키텍처 · 26x26 · none cap 15,000(train에만) ·
클래스 가중 loss · Adam 1e-3 / wd 1e-4 · Cosine 60 epoch · augmentation.
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training.train_wafermap import WaferMapCNN, WaferMapDataset, CLASS_NAMES, N_CLASSES
from scripts.a1_lot_consistency import load_cache, permutation_test, lot_lift, print_lift

SEED = 42
NONE_CAP = 15000
EPOCHS = 60
CKPT = "models/wafermap/wafermap_cnn_lotwise.pt"


def split_by_lot(lot_codes, seed=SEED, frac=(0.70, 0.15, 0.15)):
    lots = np.unique(lot_codes)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(lots))
    n1 = int(len(lots) * frac[0])
    n2 = int(len(lots) * (frac[0] + frac[1]))
    masks = []
    for sel in (perm[:n1], perm[n1:n2], perm[n2:]):
        s = np.zeros(lots.max() + 1, dtype=bool)
        s[lots[sel]] = True
        masks.append(s[lot_codes])
    return masks


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    maps_u8, y, lot_codes, lot_names = load_cache()
    maps = maps_u8.astype(np.float32) / 2.0

    m_tr, m_va, m_te = split_by_lot(lot_codes)
    n_tr_lot = len(np.unique(lot_codes[m_tr]))
    n_va_lot = len(np.unique(lot_codes[m_va]))
    n_te_lot = len(np.unique(lot_codes[m_te]))
    print(f"lot 단위 split — train lot {n_tr_lot:,} / val {n_va_lot:,} / test {n_te_lot:,}")
    print(f"  wafer: train {m_tr.sum():,} / val {m_va.sum():,} / test {m_te.sum():,}")

    # none cap — train에만 적용 (원본과 동일한 불균형 완화)
    rng = np.random.RandomState(SEED)
    tr_idx = np.where(m_tr)[0]
    none_idx = tr_idx[y[tr_idx] == 0]
    if len(none_idx) > NONE_CAP:
        drop = rng.choice(none_idx, len(none_idx) - NONE_CAP, replace=False)
        m_tr = m_tr.copy()
        m_tr[drop] = False
    print(f"  none cap 적용 후 train {m_tr.sum():,}")
    print("  train 분포:", {CLASS_NAMES[i]: int((y[m_tr] == i).sum()) for i in range(N_CLASSES)})
    print("  test  분포:", {CLASS_NAMES[i]: int((y[m_te] == i).sum()) for i in range(N_CLASSES)})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    train_ds = WaferMapDataset(maps[m_tr], y[m_tr], augment=True)
    val_ds = WaferMapDataset(maps[m_va], y[m_va], augment=False)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4)

    counts = np.bincount(y[m_tr], minlength=N_CLASSES).astype(np.float64)
    weights = counts.sum() / (N_CLASSES * np.maximum(counts, 1))
    model = WaferMapCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    from sklearn.metrics import f1_score

    best_f1, best_state = -1.0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tl, seen = 0.0, 0
        for X, yb in train_loader:
            X, yb = X.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), yb)
            loss.backward()
            optimizer.step()
            tl += loss.item() * len(X)
            seen += len(X)
        scheduler.step()
        model.eval()
        P, T = [], []
        with torch.no_grad():
            for X, yb in val_loader:
                P.append(model(X.to(device)).argmax(1).cpu())
                T.append(yb)
        P = torch.cat(P).numpy()
        T = torch.cat(T).numpy()
        f1 = float(f1_score(T, P, average="macro"))
        # 원본은 val_acc로 골랐다. 여기서는 macro F1로 고른다 (none 편향 방지)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} loss={tl/seen:.4f} "
                  f"val_acc={(P==T).mean():.4f} val_macroF1={f1:.4f}", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    print(f"\nbest val macro F1 = {best_f1:.4f}")

    # ── test lot 예측 ──
    Xte = torch.from_numpy(maps[m_te]).unsqueeze(1)
    yte = y[m_te]
    lot_te = lot_codes[m_te]
    preds = np.zeros(len(yte), dtype=np.int64)
    with torch.no_grad():
        for s in range(0, len(Xte), 4096):
            preds[s:s+4096] = model(Xte[s:s+4096].to(device)).argmax(1).cpu().numpy()
    acc = float((preds == yte).mean())
    f1_te = float(f1_score(yte, preds, average="macro"))
    print(f"test lot 성능 — acc {acc:.4f} / macro F1 {f1_te:.4f}")
    print("  예측 분포:", {CLASS_NAMES[i]: int((preds == i).sum()) for i in range(N_CLASSES)})

    out = {"val_macro_f1": best_f1, "test_acc": acc, "test_macro_f1": f1_te,
           "n_test": int(len(yte)), "n_test_lots": int(len(np.unique(lot_te)))}
    tests = []
    tests.append(permutation_test("A-1b-① 예측 패턴 — 처음 보는 lot (9종)", lot_te, preds))
    nn_ = preds != 0
    tests.append(permutation_test("A-1b-② 예측 패턴 — none 제외", lot_te[nn_], preds[nn_]))
    tests.append(permutation_test("A-1b-③ 같은 test lot의 실제 라벨 (상한)", lot_te, yte))
    out["tests"] = [t for t in tests if t]

    rows = lot_lift(lot_te, preds)
    print_lift(rows, "A-1b-④ 패턴별 lot 집중도 — 예측 기준 (처음 보는 lot)")
    out["lift_pred"] = rows
    rows_t = lot_lift(lot_te, yte)
    print_lift(rows_t, "A-1b-⑤ 같은 표 — 실제 라벨 기준 (상한)")
    out["lift_true"] = rows_t

    torch.save({"model_state_dict": best_state, "n_classes": N_CLASSES,
                "class_names": CLASS_NAMES, "map_size": 26,
                "metrics": {"val_macro_f1": round(best_f1, 4),
                            "test_acc": round(acc, 4), "test_macro_f1": round(f1_te, 4),
                            "split": "lot-wise 70/15/15"}}, CKPT)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/a1b_lotwise.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {CKPT} · outputs/a1b_lotwise.json")


if __name__ == "__main__":
    main()
