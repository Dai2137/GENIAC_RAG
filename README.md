了解しました。
以下は、**`batch_search_score.py` を中心に据えた最新版 README（完全版）** です。
誤っていた「rag_index/chunks.jsonl 参照」説明を削除し、実際の仕様（`data/result_i.jsonl` の本文をクエリに使用）に正しく訂正しています。
章・項には記号を付け、構成を視覚的に整理しています。

---

# 🧠 GENIAC-PRIZE Patent Retrieval — **Batch-first Pipeline**

本プロジェクトは、**特許文献検索・評価タスク（GENIAC PRIZE）** に対応した一括実行型パイプラインです。
メインは **`batch_search_score.py`** で、

> 💡「出願本文をクエリとしてベクトル検索 → 類似文献を取得 → AX/AY 指標でスコア評価」
> を自動で完結します。

個別実行用の `search.py` / `score_explore.py` はデバッグ・解析用の補助です。

---

## ⚙️ 0. 前提環境

| 項目     | 推奨・要件                                                                                 |
| ------ | ------------------------------------------------------------------------------------- |
| Python | 3.10 以上（3.11 推奨）                                                                      |
| OS     | macOS / Linux / Windows（PowerShell 可）                                                 |
| データ    | `data/result_1.jsonl` ～ `result_18.jsonl(.gz)`                                        |
| 評価用CSV | `data/CSV1.csv`, `data/CSV2.csv`（列：`syutugan, category(AX/AY), himotuki[, koukaibi]`） |
| API    | OpenAI互換API（ELYZA, ABEJA, rinna等）または Google Gemini                                    |

---

## 📁 1. ディレクトリ構成

```
your-project/
├─ build_index.py            # 埋め込み生成＋FAISS索引構築（初回のみ）
├─ batch_search_score.py     # ★メイン：一括検索＋評価＋保存
├─ requirements.txt
├─ data/
│   ├─ result_1.jsonl(.gz)   # 出願本文（入力）
│   ├─ ...
│   ├─ CSV1.csv              # 公式AX/AYデータ
│   └─ CSV2.csv
└─ rag_index/                # build_index.py の出力
    ├─ faiss.index
    ├─ vectors.npy
    ├─ docstore.jsonl
    ├─ fields_used.json
    └─ manifest.json
```

---

## 🧩 2. セットアップ

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

🔑 APIキー設定：

* **OpenAI互換API**

  ```bash
  export EMB_API_KEY="YOUR_KEY"
  ```
* **Gemini**

  ```bash
  export GOOGLE_API_KEY="YOUR_KEY"
  ```

---

## 🏗️ 3. 索引構築（build_index.py）

まず、検索対象（knowledge base）を作成します。
`--select` で `result_i.jsonl(.gz)` を番号指定可能。

### 3.1 OpenAI互換APIで作成

```bash
python build_index.py \
  --data_dir ./data \
  --out_dir  ./rag_index \
  --provider openai_compat \
  --api_base https://api.example.com/v1 \
  --emb_model embedding-japanese-v1 \
  --api_key_env EMB_API_KEY \
  --select "1,3-5,12" \
  --chunk_size 1200 --chunk_overlap 200 \
  --limit_docs 10
```

### 3.2 Geminiで作成

```bash
python build_index.py \
  --data_dir ./data \
  --out_dir  ./rag_index \
  --provider gemini \
  --emb_model models/embedding-001 \
  --api_key_env GOOGLE_API_KEY \
  --select "1,3-5,12" \
  --chunk_size 1200 --chunk_overlap 200 \
  --limit_docs 10
```

📌 **ポイント**

* `.jsonl` と `.jsonl.gz` の両方に対応
* `--select` 形式： `"1,3-5,12"` → 1,3,4,5,12 を処理
* 既存のインデックスは再生成されます（再構築仕様）

出力例：

```
rag_index/
├─ faiss.index
├─ vectors.npy
├─ docstore.jsonl
├─ manifest.json
└─ fields_used.json
```

---

## 🚀 4. 一括検索＋評価（batch_search_score.py）

`batch_search_score.py` がこのリポジトリのメインスクリプトです。

### 🔍 クエリ生成の正確な仕様

> **検索クエリは、`data/result_i.jsonl(.gz)` に含まれる出願本文から直接生成されます。**
> 各出願の `"title"`, `"abstract"`, `"description"`, `"claims"` を結合し、
> それをクエリとして使用します。
> `rag_index` 内のデータは検索対象であり、クエリ生成には使用しません。

---

### 4.1 OpenAI互換APIで実行

```bash
python batch_search_score.py \
  --data_dir ./data \
  --select "1,3-5,12" \
  --limit_docs 0 \
  --index_dir ./rag_index \
  --truth ./data/CSV1.csv ./data/CSV2.csv \
  --k 50 --mMax 10 --P 0.8 \
  --provider openai_compat \
  --api_base https://api.example.com/v1 \
  --api_key_env EMB_API_KEY \
  --emb_model embedding-japanese-v1
```

---

### 4.2 Geminiで実行

```bash
python batch_search_score.py \
  --data_dir ./data \
  --select "1-18" \
  --limit_docs 0 \
  --index_dir ./rag_index \
  --truth ./data/CSV1.csv ./data/CSV2.csv \
  --k 50 --mMax 10 --P 0.8 \
  --provider gemini \
  --api_key_env GOOGLE_API_KEY \
  --emb_model models/embedding-001
```

---

### ⚖️ 主な引数（評価設定）

| 引数             | 意味                           |
| -------------- | ---------------------------- |
| `--select`     | 使用する出願ファイル番号 (`"1,3-5,12"`)  |
| `--k`          | 上位 K 件を評価に使用                 |
| `--mMax`       | AY 最大ヒット件数（上限）               |
| `--P`          | AX/AY の重み係数（0.0〜1.0）         |
| `--limit_docs` | クエリ件数上限（0=全件）                |
| `--index_dir`  | 検索対象インデックス（`rag_index`）      |
| `--truth`      | 公式CSV（CSV1/CSV2）             |
| `--provider`   | `openai_compat` または `gemini` |

---

## 📊 5. 出力内容

`batch_search_score.py` 実行後、自動で以下のファイルが生成されます。

```
score_results/
├─ summary.csv                  # 出願ごとのスコア集計
├─ JP2012239158A_result.json    # 個別評価（AX/AY命中詳細など）
├─ JP2022123456A_result.json
└─ ...
retrieved_pairs.csv             # 検索結果の (query_id, knowledge_id) ペア
```

| ファイル                  | 内容                                                     |
| --------------------- | ------------------------------------------------------ |
| `summary.csv`         | 各出願の `ax_hit`, `ay_hit`, `score_raw`, `score_scaled` 等 |
| `_result.json`        | 個票：ヒット順位、スコア内訳、対象文献情報                                  |
| `retrieved_pairs.csv` | クエリ ↔ 類似文献ペアの生データ                                      |

---

## 🔄 6. 一連の推奨フロー

| 手順       | コマンド例                       | 説明            |
| -------- | --------------------------- | ------------- |
| 🏗️ 索引構築 | `build_index.py`            | 1回だけ実行すればOK   |
| 🚀 一括評価  | `batch_search_score.py`     | 出願群をまとめて検索＋採点 |
| 📈 分析    | `score_results/summary.csv` | スコア分布・順位分析に活用 |

💡 個別デバッグが必要な場合のみ `search.py` / `score_explore.py` を使用。

---

## 🧠 7. 評価ロジック概要

* **AXヒット（同カテゴリ内一致）**：主要スコア要素
* **AYヒット（関連出願群内一致）**：部分加点要素
* **mMax / P** パラメータで、上限件数と寄与比を調整
* スコアは `score_raw` → `score_scaled` に正規化して保存

---

## 🩺 8. トラブルシューティング

| 症状                      | 原因・対策                                                     |
| ----------------------- | --------------------------------------------------------- |
| ❌ `no valid docs found` | `--select` の指定番号が存在しない。 `"1,3-5"` のように確認                  |
| ⚠️ `API error 401/429`  | APIキー設定またはレート上限。`--rpm`を下げて再試行                            |
| 🈳 スコアが0ばかり             | 評価CSVの `syutugan` と `result_i` 内の出願番号表記を統一                |
| 🧩 索引が空                 | `build_index.py` の `fields_used.json` を確認し、text抽出項目を再チェック |

---

## 💡 9. クイックスタート（最短実行例）

### OpenAI互換API 版

```bash
# 1) セットアップ
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export EMB_API_KEY="YOUR_KEY"

# 2) 索引作成
python build_index.py \
  --data_dir ./data --out_dir ./rag_index \
  --provider openai_compat \
  --api_base https://api.example.com/v1 \
  --emb_model embedding-japanese-v1 \
  --api_key_env EMB_API_KEY \
  --select "1,2" \
  --limit_docs 10

# 3) 一括検索＋評価
python batch_search_score.py \
  --data_dir ./data \
  --select "1,2" \
  --index_dir ./rag_index \
  --truth ./data/CSV1.csv ./data/CSV2.csv \
  --k 50 --mMax 10 --P 0.8 \
  --provider openai_compat \
  --api_base https://api.example.com/v1 \
  --api_key_env EMB_API_KEY \
  --emb_model embedding-japanese-v1 \
  --limit_docs 10
```

### Gemini 版

```bash
export GOOGLE_API_KEY="YOUR_KEY"
python build_index.py \
  --data_dir ./data --out_dir ./rag_index \
  --provider gemini --emb_model models/embedding-001 \
  --api_key_env GOOGLE_API_KEY \
  --limit_docs 10
python batch_search_score.py \
  --data_dir ./data --select "1-18" \
  --index_dir ./rag_index \
  --truth ./data/CSV1.csv ./data/CSV2.csv \
  --provider gemini --api_key_env GOOGLE_API_KEY \
  --emb_model models/embedding-001 \
  --limit_docs 10
```

---

## 🧾 10. 開発メモ

* **FAISS**：`IndexFlatIP`（Cosine相当）を使用。`--use_gpu_faiss` でGPU転送可。
* **ベクトル正規化**：L2正規化して登録。
* **再現性**：乱数種 `--seed`（既定42）固定。
* **ログ出力**：全主要処理で標準出力に進行表示（tqdm＋INFOログ）。
* **出力ディレクトリ**：`score_results/` 以下にすべて保存されるため、結果が失われません。

---

## 🧭 11. よくある質問（FAQ）

| 質問                     | 回答                                                       |
| ---------------------- | -------------------------------------------------------- |
| Q. `--select` の指定方法は？  | `"1,3-5,12"` のようにカンマ区切り＋範囲指定可能。                          |
| Q. 公式CSVが文字化けします。      | UTF-8 / CP932 自動判定で読込み。列名の欠落に注意。                         |
| Q. 評価は親文献単位？           | はい。チャンクを親IDで集約後、上位K件で評価。                                 |
| Q. batchでGeminiを使うと遅い？ | Geminiは一括APIが無いため、内部で逐次呼び出しを行います。`--rpm`/`--batch`で調整可能。 |

---

### 🏁 総括

> * `build_index.py`：検索対象（FAISSインデックス）を作成
> * `batch_search_score.py`：本文クエリを用いた一括検索＋AX/AY評価
> * `score_results/`：結果はすべて保存・可視化可能

この2ステップのみで、GENIAC PRIZE 形式の検索・評価実験が完結します。
（`search.py` と `score_explore.py` は内部処理の分離版として残されています）

---