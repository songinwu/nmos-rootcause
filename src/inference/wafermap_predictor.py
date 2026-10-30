"""
Wafer Defect Map 분류 추론 클래스.
입력: 26×26 또는 평탄화된 676 포인트 웨이퍼 맵.
      값 규약은 train_wafermap.MAP_ENCODING (0=배경 / 0.5=정상 / 1.0=결함).
출력: 결함 유형 분류 + 확률 분포.
"""

import os
import numpy as np
import torch
import torch.nn as nn


MAP_SIZE    = 26
N_CLASSES   = 9
CLASS_NAMES = ["none", "center", "donut", "edge-loc", "edge-ring",
               "loc", "near-full", "random", "scratch"]


def _select_torch_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.empty(1, device="cuda") + 1
        return torch.device("cuda")
    except Exception:
        return torch.device("cpu")


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


class WaferMapPredictor:
    def __init__(self, model_path: str = "models/wafermap/wafermap_cnn.pt"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Wafermap model not found: {model_path}")

        self.device = _select_torch_device()
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

        self.n_classes   = ckpt.get("n_classes", N_CLASSES)
        self.class_names = ckpt.get("class_names", CLASS_NAMES)
        self.map_size    = ckpt.get("map_size", MAP_SIZE)
        self.metrics     = ckpt.get("metrics", {})

        self.model = WaferMapCNN(self.n_classes).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def _to_tensor(self, data) -> torch.Tensor:
        arr = np.array(data, dtype=np.float32)
        if arr.ndim == 1:
            if len(arr) == self.map_size ** 2:
                arr = arr.reshape(self.map_size, self.map_size)
            else:
                raise ValueError(f"1D input must have {self.map_size**2} elements, got {len(arr)}")
        if arr.ndim == 2:
            arr = arr[None, None, :, :]         # (1,1,H,W)
        elif arr.ndim == 3:
            arr = arr[None, :, :, :]
        return torch.tensor(arr).to(self.device)

    @torch.no_grad()
    def predict(self, wafer_map) -> dict:
        """
        Parameters
        ----------
        wafer_map : list[list] | list | np.ndarray
            26×26 맵 또는 676 element 1D 배열.

            값 규약은 학습과 같아야 한다 (train_wafermap.MAP_ENCODING):
                0.0 = 배경(웨이퍼 원판 밖)
                0.5 = 정상 다이
                1.0 = 결함 다이
            이진(0/1)이 아니다. 0/1로 넣으면 정상 다이가 배경으로 읽혀
            분류가 조용히 어긋난다.

        Returns
        -------
        dict with predicted class, probability distribution, and wafer map
        """
        try:
            X = self._to_tensor(wafer_map)
            logits = self.model(X)
        except RuntimeError as e:
            if self.device.type != "cuda" or "CUDA" not in str(e).upper():
                raise
            self.device = torch.device("cpu")
            self.model.to(self.device)
            X = self._to_tensor(wafer_map)
            logits = self.model(X)

        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx  = int(np.argmax(probs))
        pred_name = self.class_names[pred_idx]

        # 정상 여부
        is_defective = pred_name != "none"

        return {
            "module": "wafer_defect_map",
            "prediction": {
                "defect_type": pred_name,
                "defect_index": pred_idx,
                "is_defective": is_defective,
                "confidence_pct": round(float(probs[pred_idx]) * 100, 2),
            },
            "probability_distribution": {
                name: round(float(p) * 100, 2)
                for name, p in zip(self.class_names, probs)
            },
            "role": self.ROLE,
            "model_info": {
                "type": "2D CNN",
                "n_classes": self.n_classes,
                "map_size": f"{self.map_size}×{self.map_size}",
                "val_accuracy_pct": round(self.metrics.get("val_accuracy", 0) * 100, 1),
                "measured_on": self.MEASUREMENT_CONDITION,
            },
        }

    # 2026-08-17 검증(설계 문서 4-13절). 보고해온 88.7% / macro F1 0.837은
    # **none을 15,000장으로 깎은 셋**에서 잰 값이다. 학습에 쓰지 않은 138,509장
    # 전부(실제 라인처럼 none 97.24%)로 재평가하니 정확도는 0.929로 **오히려 오르고**
    # macro F1은 0.558로 무너졌다 — scratch는 경보 100건 중 94건이 오경보다.
    # 그래서 이 모듈은 분류기가 아니라 **1차 스크리너**로 표기한다.
    ROLE = {
        "kind": "screener",
        "not_a_classifier": (
            "이 모듈은 결함 유형을 확정하는 분류기가 아니라, 사람이 볼 큐를 만드는 "
            "1차 스크리너다. 실제 라인 분포에서 소수 클래스 정밀도가 무너지기 때문이다."
        ),
        "enrichment": {
            "defect_rate_before_pct": 2.8,
            "defect_rate_after_pct": 29.8,
            "enrichment_x": 10.8,
            "recall": 0.877,
            "note": "재현율이 유지되므로 좁히는 데는 쓸 수 있다 — 확정하는 데는 쓸 수 없다.",
        },
    }

    MEASUREMENT_CONDITION = {
        "reported_metric_basis": "none을 15,000장으로 cap한 셋의 validation",
        "real_distribution": {
            "n": 138509,
            "none_share_pct": 97.24,
            "accuracy": 0.929,
            "macro_f1": 0.558,
            "macro_precision": 0.467,
            "macro_recall": 0.877,
            "worst_class": {"name": "scratch", "precision": 0.040,
                            "false_alarm_share_pct": 94.2},
        },
        "warning": (
            "🔴 실제 분포에서는 정확도가 0.929로 오르지만 macro F1은 0.558로 반토막이다. "
            "정확도만 말하면 거짓이 된다."
        ),
        "prior_work": (
            "같은 구성(none 15k cap)으로 학습한 선행 연구도 동일 패턴이며"
            "(balanced 0.913 → official 0.703) 약한 클래스 위치(loc·scratch)까지 같다 "
            "— 모델 설계 실수가 아니라 데이터셋의 난이도 구조다."
        ),
    }
