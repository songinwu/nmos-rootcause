#!/bin/bash
# WM-811K Wafer Map Dataset 다운로드 스크립트
#
# 사전 준비:
#   1. https://www.kaggle.com 로그인 → 우측 상단 프로필 → Settings → API → "Create New Token"
#   2. 다운로드된 kaggle.json을 아래 경로 중 하나에 배치:
#        ~/.config/kaggle/kaggle.json  (신버전 kaggle CLI 기본)
#        ~/.kaggle/kaggle.json         (구버전 호환)
#   3. chmod 600 <위 파일 경로>
#
# 실행:
#   bash scripts/download_wm811k.sh

set -e

DEST="data/raw/wm811k"
mkdir -p "$DEST"

echo "[1] Kaggle API key 확인"
KAGGLE_JSON=""
if [ -f ~/.config/kaggle/kaggle.json ]; then
    KAGGLE_JSON=~/.config/kaggle/kaggle.json
elif [ -f ~/.kaggle/kaggle.json ]; then
    KAGGLE_JSON=~/.kaggle/kaggle.json
fi

if [ -z "$KAGGLE_JSON" ]; then
    echo "ERROR: kaggle.json 없음"
    echo ""
    echo "  1. https://www.kaggle.com 로그인"
    echo "  2. 우측 상단 프로필 → Settings → API → 'Create New Token'"
    echo "  3. 다운로드된 kaggle.json 을 아래 경로에 배치:"
    echo "       mkdir -p ~/.config/kaggle && mv ~/Downloads/kaggle.json ~/.config/kaggle/"
    echo "       chmod 600 ~/.config/kaggle/kaggle.json"
    echo "  4. 이 스크립트 재실행"
    exit 1
fi
chmod 600 "$KAGGLE_JSON"
echo "    API key: $KAGGLE_JSON"

echo "[2] WM-811K 다운로드 (qingyi310/wm-811k-wafermap)"
kaggle datasets download -d qingyi310/wm-811k-wafermap -p "$DEST" --unzip

echo "[3] pkl 파일 확인"
ls -lh "$DEST/"
echo ""

if ls "$DEST"/*.pkl 1>/dev/null 2>&1; then
    echo "[OK] pkl 파일 발견"
    PKL=$(ls "$DEST"/*.pkl | head -1)
    if [ "$PKL" != "$DEST/WM811K.pkl" ]; then
        mv "$PKL" "$DEST/WM811K.pkl"   # a1_lot_consistency.py가 이 이름을 찾는다
        echo "  이름 변경 → WM811K.pkl"
    fi
    echo ""
    echo "[DONE] 완료. 이제 재학습:"
    echo "  python -m src.training.train_wafermap"
else
    echo "[WARN] pkl 파일 없음"
    echo "  수동으로 WM811K.pkl 을 $DEST/ 에 배치 후:"
    echo "  python -m src.training.train_wafermap"
fi
