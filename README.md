# インストール手順メモ

## 仮想環境作成（任意）

```bash
python -m venv .venv
```

## 有効化

# Windows:
source .venv/Scripts/activate
# macOS / Linux:
source .venv/bin/activate

## 依存インストール

```bash
pip install -r requirements.txt
```


# Colabのセットアップセル
!nvidia-smi  # GPU が見えるか確認（T4/A100などが出ればOK）

# 依存をインストール（ColabはCUDA版PyTorchが既に入っています）
!pip -q install faiss-gpu sentence-transformers ujson pandas tqdm rank_bm25

Driveを使う場合
from google.colab import drive
drive.mount('/content/drive')

# 使い方（Colabセルで）：
Drive の JSONL を使う例
!python build_index.py \
  --data_dir "/content/drive/MyDrive/Colab Notebooks/GENIAC" \
  --out_dir "/content/rag_index" \
  --model intfloat/multilingual-e5-small \
  --batch 256 --max_seq 256 --use_gpu_faiss

!python search.py --index_dir "/content/rag_index" \
  --q "リチウムイオン二次電池 電解液添加剤" --k 10 --group_by_parent

!python score_explore.py \
  --truth "/content/drive/MyDrive/Colab Notebooks/GENIAC/ax_ay_truth.csv" \
  --alpha JP20XXXXXXXA \
  --retrieved_json "/content/retrieved.json"
