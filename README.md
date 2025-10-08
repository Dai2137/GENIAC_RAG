# 🚀 インストール手順メモ

---

## 🧩 1. 仮想環境の作成（任意）

```bash
python -m venv .venv
````

### 有効化

#### Windows

```bash
.venv\Scripts\activate
```

#### macOS / Linux

```bash
source .venv/bin/activate
```

---

## 📦 2. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

---

## 💻 3. Google Colab でのセットアップ

> Colab（T4 / CUDA12.x）で動かす場合はこちらをセルごとにコピーして実行。

### 🧹 (1) 競合しやすい既存パッケージの削除

```bash
!pip -q uninstall -y \
  faiss-gpu-cu12 faiss-gpu faiss-cpu \
  opencv-python opencv-contrib-python opencv-python-headless \
  spacy thinc tensorflow fastai dopamine-rl \
  pandas
```

### 🧱 (2) NumPy / pandas / FAISS / OpenCV / 主要ツールのセットアップ

```bash
# NumPy（FAISS互換バージョン）
!pip -q install --no-cache-dir "numpy==1.26.4"

# pandas（Colab互換）
!pip -q install --no-cache-dir "pandas==2.2.2"

# FAISS (GPU, CUDA 12.x)
!pip -q install --no-cache-dir "faiss-gpu-cu12==1.12.0"

# OpenCV（必要な場合のみ。GUI不要なら headless が安定）
!pip -q install --no-cache-dir "opencv-python-headless==4.9.0.80"

# その他ツール
!pip -q install --no-cache-dir ujson tqdm rank_bm25 sentence-transformers
```

> ⚠️ **実行後に必ず「ランタイムを再起動」してください。**
> 手動操作：メニュー → 「ランタイム」→「ランタイムを再起動」

---

### ✅ (3) 動作確認セル

```python
import numpy as np, pandas as pd, torch, faiss

print("NumPy:", np.__version__)             # -> 1.26.4
print("pandas:", pd.__version__)            # -> 2.2.2
print("CUDA (torch):", torch.version.cuda)  # -> 12.6 ならOK
print("FAISS:", faiss.__version__, "GPUs:", faiss.get_num_gpus())

# OpenCV を入れた場合のみ
try:
    import cv2
    print("OpenCV:", cv2.__version__)       # -> 4.9.0.80
except Exception as e:
    print("OpenCV not installed (OK):", e)
```

---

## 📂 4. Google Drive を使う場合

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 🔍 5. 使い方（Colab セルでの実行例）

### 📘 (1) インデックス構築

```bash
!python build_index.py \
  --data_dir "/content/drive/MyDrive/Colab Notebooks/GENIAC" \
  --out_dir "/content/rag_index" \
  --model intfloat/multilingual-e5-small \
  --batch 256 --max_seq 256 --use_gpu_faiss
```

### 🔎 (2) 検索実行

```bash
!python search.py \
  --index_dir "/content/rag_index" \
  --q "リチウムイオン二次電池 電解液添加剤" \
  --k 10 --group_by_parent
```

### 📊 (3) 検索結果のスコア評価

```bash
!python score_explore.py \
  --truth "/content/drive/MyDrive/Colab Notebooks/GENIAC/ax_ay_truth.csv" \
  --alpha JP20XXXXXXXA \
  --retrieved_json "/content/retrieved.json"
```

---

## 🧾 まとめ

| 区分           | 目的                 | 主なコマンド                                            |
| ------------ | ------------------ | ------------------------------------------------- |
| 仮想環境構築       | ローカル環境を分離          | `python -m venv .venv`                            |
| Colab セットアップ | 依存を統一              | 上記 Colab セルを実行                                    |
| 動作確認         | ライブラリのバージョン確認      | NumPy / FAISS / CUDA                              |
| 実行           | インデックス構築 → 検索 → 評価 | `build_index.py`, `search.py`, `score_explore.py` |