"""
Wafer Defect Map 분류 — 2D CNN 학습 스크립트.

데이터 우선순위:
  1. data/raw/wm811k/*.pkl          (WM-811K 원본, MIR Lab / Kaggle LSWMD)
  2. data/raw/wm811k/WM811k_Dataset (JPG 이미지 폴더, 902장)
  3. 합성 데이터 자동 생성

결함 패턴 9종:
  0=none, 1=center, 2=donut, 3=edge-loc, 4=edge-ring,
  5=loc, 6=near-full, 7=random, 8=scratch

WM-811K pkl 처리:
  - waferMap 값: 0=배경(웨이퍼 밖), 1=정상 다이, 2=결함 다이 → /2.0 정규화
  - 크기 제각각 → PIL NEAREST로 26×26 resize
  - 극심한 클래스 불균형 (none이 85%) → none 캡 + 클래스 가중 loss + 회전/반전 augmentation
"""

import os, glob, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

MAP_SIZE = 26
N_CLASSES = 9
CLASS_NAMES = ["none", "center", "donut", "edge-loc", "edge-ring",
               "loc", "near-full", "random", "scratch"]
OUT_DIR   = "models/wafermap"
OUT_PATH  = os.path.join(OUT_DIR, "wafermap_cnn.pt")

NONE_CAP  = 15000   # 'none' 클래스 최대 샘플 수 (원본 약 14.7만 장)
SEED      = 42


# ──────────────────────────────────────────────
# Synthetic data generator (fallback + 보강용)
# ──────────────────────────────────────────────

def _circle_mask(size, cx, cy, r):
    Y, X = np.ogrid[:size, :size]
    return (X - cx)**2 + (Y - cy)**2 <= r**2

def _ring_mask(size, cx, cy, r_in, r_out):
    return _circle_mask(size, cx, cy, r_out) & ~_circle_mask(size, cx, cy, r_in)

def _edge_loc(size):
    m = np.zeros((size, size), dtype=np.float32)
    side = random.randint(0, 3)
    w = size // 5
    if side == 0: m[:w, :] = 1
    elif side == 1: m[-w:, :] = 1
    elif side == 2: m[:, :w] = 1
    else: m[:, -w:] = 1
    return m

def _loc_cluster(size):
    cy = random.randint(size//4, 3*size//4)
    cx = random.randint(size//4, 3*size//4)
    r  = random.randint(2, size//6)
    return _circle_mask(size, cx, cy, r).astype(np.float32)

def _scratch(size):
    m = np.zeros((size, size), dtype=np.float32)
    angle = random.choice([0, 1, 2, 3])
    if angle == 0:
        r = random.randint(2, size-3)
        m[r:r+1, :] = 1
    elif angle == 1:
        c = random.randint(2, size-3)
        m[:, c:c+1] = 1
    elif angle == 2:
        for i in range(size): m[i, i] = 1
    else:
        for i in range(size): m[i, size-1-i] = 1
    return m

def make_synthetic(n_per_class=400, size=MAP_SIZE, noise=0.08):
    maps, labels = [], []
    c = size // 2
    generators = [
        lambda: np.zeros((size, size), dtype=np.float32),
        lambda: _circle_mask(size, c, c, size//6).astype(np.float32),
        lambda: _ring_mask(size, c, c, size//6, size//3).astype(np.float32),
        lambda: _edge_loc(size),
        lambda: _ring_mask(size, c, c, size//2 - 4, size//2).astype(np.float32),
        lambda: _loc_cluster(size),
        lambda: (_circle_mask(size, c, c, size//2 - 1)).astype(np.float32),
        lambda: (np.random.rand(size, size) < 0.15).astype(np.float32),
        lambda: _scratch(size),
    ]
    for label, gen in enumerate(generators):
        for _ in range(n_per_class):
            m = np.clip(gen() + np.random.randn(size, size) * noise, 0, 1)
            maps.append(m.astype(np.float32))
            labels.append(label)
    return np.array(maps, dtype=np.float32), np.array(labels, dtype=np.int64)


# ──────────────────────────────────────────────
# Dataset (train 시 회전/반전 augmentation)
# ──────────────────────────────────────────────

class WaferMapDataset(Dataset):
    def __init__(self, maps, labels, augment=False):
        self.maps    = torch.tensor(maps[:, None, :, :])   # (N,1,H,W)
        self.labels  = torch.tensor(labels)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x, y = self.maps[i], self.labels[i]
        if self.augment:
            k = random.randint(0, 3)
            if k:
                x = torch.rot90(x, k, dims=(1, 2))
            if random.random() < 0.5:
                x = torch.flip(x, dims=(2,))
        return x, y


# ──────────────────────────────────────────────
# 2D CNN model  (추론 클래스 wafermap_predictor.py와 구조 동일 유지 필수)
# ──────────────────────────────────────────────

class WaferMapCNN(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ──────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────

# 웨이퍼 맵 값의 의미 — 이 표가 정본이다.
# WM-811K 원본은 0/1/2 정수이고, 학습 입력은 그것을 2로 나눈 값이다.
# 이진(0/1)이 아니라는 점이 중요하다. 0/1로 만들면 정상 다이가 배경으로 읽힌다.
MAP_ENCODING = {
    "background": 0.0,   # 웨이퍼 원판 밖 (원본 0)
    "good": 0.5,         # 정상 다이     (원본 1)
    "defect": 1.0,       # 결함 다이     (원본 2)
}
MAP_ENCODING_DOC = (
    "0=배경(웨이퍼 밖), 0.5=정상 다이, 1.0=결함 다이 — 이진이 아니라 3값이다"
)


def _resize_map(wm: np.ndarray) -> np.ndarray:
    """가변 크기 웨이퍼 맵 → 26×26. 값 0/1/2 유지 위해 NEAREST.

    반환값은 MAP_ENCODING 규약을 따른다 (배경 0.0 / 정상 0.5 / 결함 1.0).
    """
    from PIL import Image
    img = Image.fromarray(wm.astype(np.uint8), mode="L")
    img = img.resize((MAP_SIZE, MAP_SIZE), Image.NEAREST)
    return np.array(img, dtype=np.float32) / 2.0   # 0=배경, 0.5=정상, 1.0=결함


def load_from_pkl(pkl_path: str) -> tuple:
    """WM-811K pkl 로드 — 라벨 있는 맵 전체 사용 (크기 무관 resize). npz 캐시 지원."""
    import pandas as pd

    cache_path = "data/processed/wm811k_resized26.npz"
    if os.path.exists(cache_path):
        print(f"npz 캐시 로드: {cache_path}")
        z = np.load(cache_path)
        return z["maps"], z["labels"]

    print(f"WM-811K pkl 로드 중: {pkl_path} (수 분 소요) ...")
    df = pd.read_pickle(pkl_path)
    print(f"  전체 {len(df)} rows")

    label_map = {n: i for i, n in enumerate(CLASS_NAMES)}

    def norm_label(ft):
        # failureType 형태 다양: np.ndarray([['Edge-Loc']]) / [] / 'none'
        while isinstance(ft, (list, np.ndarray)):
            if len(ft) == 0:
                return None
            ft = ft[0]
        s = str(ft).strip().lower()      # 'Edge-Loc' → 'edge-loc' (하이픈 유지)
        return s if s in label_map else None

    labels_raw = df["failureType"].map(norm_label)
    mask = labels_raw.notna()
    df_lab = df[mask]
    labels_str = labels_raw[mask]
    print(f"  라벨 있는 맵: {len(df_lab)}")

    # 'none' 클래스 캡 (불균형 완화 + 학습 시간)
    rng = np.random.RandomState(SEED)
    keep_idx = []
    for cls in CLASS_NAMES:
        idx = labels_str[labels_str == cls].index.to_numpy()
        if cls == "none" and len(idx) > NONE_CAP:
            idx = rng.choice(idx, NONE_CAP, replace=False)
        keep_idx.append(idx)
    keep_idx = np.concatenate(keep_idx)

    maps_list, labels_list = [], []
    for n, i in enumerate(keep_idx):
        wm = df_lab.at[i, "waferMap"]
        maps_list.append(_resize_map(np.asarray(wm)))
        labels_list.append(label_map[labels_str.at[i]])
        if (n + 1) % 10000 == 0:
            print(f"  resize {n + 1}/{len(keep_idx)}")

    maps   = np.array(maps_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int64)
    print("  클래스별:", {CLASS_NAMES[i]: int((labels == i).sum()) for i in range(N_CLASSES)})

    os.makedirs("data/processed", exist_ok=True)
    np.savez_compressed(cache_path, maps=maps, labels=labels)
    print(f"  npz 캐시 저장: {cache_path}")
    return maps, labels


_IMG_FOLDER_MAP = {
    "center": "Center", "donut": "Donut", "edge-loc": "Edge Local",
    "edge-ring": "Edge Ring", "loc": "Local", "near-full": "near full",
    "random": "random", "scratch": "Scratch", "none": "none",
}
_FOLDER_TO_CLASS = {v.lower(): k for k, v in _IMG_FOLDER_MAP.items()}


def _load_from_img_dir(img_root: str) -> tuple:
    """WM811k_Dataset/ 폴더 구조에서 JPG 이미지 로드."""
    from PIL import Image
    label_map = {n: i for i, n in enumerate(CLASS_NAMES)}
    maps_list, labels_list = [], []
    for folder in os.listdir(img_root):
        folder_path = os.path.join(img_root, folder)
        if not os.path.isdir(folder_path):
            continue
        class_key = _FOLDER_TO_CLASS.get(folder.lower())
        if class_key is None:
            continue
        lbl = label_map[class_key]
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith((".jpg", ".png")):
                continue
            try:
                img = Image.open(os.path.join(folder_path, fname)).convert("L")
                img = img.resize((MAP_SIZE, MAP_SIZE), Image.NEAREST)
                maps_list.append(np.array(img, dtype=np.float32) / 255.0)
                labels_list.append(lbl)
            except Exception:
                pass
    return np.array(maps_list, dtype=np.float32), np.array(labels_list, dtype=np.int64)


def _find_pkl() -> str:
    for pattern in ["data/raw/wm811k/LSWMD.pkl",
                    "data/raw/wm811k/WM811K.pkl",
                    "data/raw/wm811k/**/*.pkl"]:
        hits = glob.glob(pattern, recursive=True)
        if hits:
            return hits[0]
    return ""


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pkl_path  = _find_pkl()
    wm_imgdir = "data/raw/wm811k/WM811k_Dataset"
    data_source = "synthetic"

    if pkl_path:
        maps, labels = load_from_pkl(pkl_path)
        data_source = f"wm811k_pkl ({os.path.basename(pkl_path)})"
    elif os.path.exists(wm_imgdir):
        print(f"WM-811K 이미지 폴더 로드 중: {wm_imgdir}")
        maps, labels = _load_from_img_dir(wm_imgdir)
        data_source = "wm811k_images"
        if len(maps) > 50:
            real_per_class = np.bincount(labels, minlength=N_CLASSES)
            need_synth = max(0, 200 - real_per_class.min())
            if need_synth > 0:
                s_maps, s_labels = make_synthetic(n_per_class=need_synth)
                maps   = np.concatenate([maps, s_maps])
                labels = np.concatenate([labels, s_labels])
        else:
            maps, labels = make_synthetic(n_per_class=600)
            data_source = "synthetic"
    else:
        print("합성 데이터 생성 중 (WM-811K 없음) ...")
        maps, labels = make_synthetic(n_per_class=600)

    print(f"데이터 소스: {data_source}, 총 {len(maps)} samples")

    # ── Stratified split ──
    from sklearn.model_selection import train_test_split
    idx_train, idx_val = train_test_split(
        np.arange(len(labels)), test_size=0.15,
        stratify=labels, random_state=SEED,
    )
    train_ds = WaferMapDataset(maps[idx_train], labels[idx_train], augment=True)
    val_ds   = WaferMapDataset(maps[idx_val],   labels[idx_val],   augment=False)
    print(f"Train {len(train_ds)} / Val {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=2)

    # ── 클래스 가중 loss (불균형 대응) ──
    counts = np.bincount(labels[idx_train], minlength=N_CLASSES).astype(np.float64)
    weights = counts.sum() / (N_CLASSES * np.maximum(counts, 1))
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    print("클래스 가중치:", np.round(weights, 2).tolist())

    model = WaferMapCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    EPOCHS = 60
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc, best_macro_f1, best_state = 0.0, 0.0, None
    best_preds, best_true = None, None

    from sklearn.metrics import f1_score

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, n_seen = 0.0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_b)
            n_seen += len(X_b)
        scheduler.step()

        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                preds = model(X_b.to(device)).argmax(1).cpu()
                all_preds.append(preds)
                all_true.append(y_b)
        all_preds = torch.cat(all_preds).numpy()
        all_true  = torch.cat(all_true).numpy()
        val_acc  = float((all_preds == all_true).mean())
        macro_f1 = float(f1_score(all_true, all_preds, average="macro"))

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_macro_f1 = macro_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_preds, best_true = all_preds.copy(), all_true.copy()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={train_loss/n_seen:.4f}  "
                  f"val_acc={val_acc:.4f}  macro_f1={macro_f1:.4f}")

    per_class_acc = {}
    for i, name in enumerate(CLASS_NAMES):
        m = best_true == i
        per_class_acc[name] = round(float((best_preds[m] == i).mean()), 4) if m.sum() else None

    print(f"\nBest val accuracy: {best_val_acc:.4f}  macro F1: {best_macro_f1:.4f}")
    print("클래스별 acc:", per_class_acc)

    checkpoint = {
        "model_state_dict": best_state,
        "n_classes": N_CLASSES,
        "class_names": CLASS_NAMES,
        "map_size": MAP_SIZE,
        "metrics": {
            "val_accuracy": round(best_val_acc, 4),
            "macro_f1": round(best_macro_f1, 4),
            "data_source": data_source,
            "n_samples": int(len(maps)),
        },
    }
    torch.save(checkpoint, OUT_PATH)
    print(f"Model saved → {OUT_PATH}")

    metrics = {
        "wafermap_val_accuracy": round(best_val_acc, 4),
        "macro_f1": round(best_macro_f1, 4),
        "per_class_accuracy": per_class_acc,
        "data_source": data_source,
        "n_samples": int(len(maps)),
        "n_classes": N_CLASSES,
    }
    with open("outputs/metrics/wafermap_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
