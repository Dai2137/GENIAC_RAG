了解しました。
以下に、**`build_index.py` ＆ `batch_search_score.py` の実際の実装仕様に完全準拠した README（最新版・完全版）** を示します。
これ1本で「埋め込み → インデックス構築 → 検索・評価」の全体構造と内部処理が把握できるように構成しています。

---

# 🧠 GENIAC-PRIZE Patent Retrieval — 完全実装ドキュメント

本プロジェクトは、**特許文献検索タスク（GENIAC PRIZE）** に対応した一括実行型 RAG 検索評価フレームワークです。

主な流れ：

> 出願本文をクエリに → 既存文献のチャンク群を対象にベクトル検索 →
> 類似文献を取得 → 公式 AX/AY データでスコア評価

---

## ⚙️ 0. 実行環境

| 項目     | 推奨環境                                                 |
| ------ | ---------------------------------------------------- |
| Python | 3.10 以上                                              |
| OS     | macOS / Linux / Windows (PowerShell対応)               |
| GPU    | 不要（FAISS CPU版対応）                                     |
| API    | Google Gemini または OpenAI互換Embedding API              |
| データ    | `data/result_1.jsonl` ～ `result_18.jsonl(.gz)`（特許文献） |
| 評価CSV  | `CSV1.csv`, `CSV2.csv`（AX/AY対応データ）                   |

---

## 📁 1. ディレクトリ構成

```
project/
├─ build_index.py            # 既存文献のベクトル化＆FAISS構築
├─ batch_search_score.py     # クエリ検索＋評価（メイン）
├─ requirements.txt
├─ data/
│   ├─ result_1.jsonl(.gz)   # 特許文献データ
│   ├─ CSV1.csv, CSV2.csv    # AX/AYデータ
└─ rag_index/                # インデックス出力
    ├─ faiss.index           # FAISS本体
    ├─ vectors.npy           # 各チャンクのベクトル
    ├─ docstore.jsonl        # 各チャンクの原文・親ID情報
    ├─ emb_ids.npy, parent_ids.npy
    └─ manifest.json         # 実行パラメータ
```

---

## 🧩 2. 埋め込み（build_index.py）

### 🧭 処理概要

1. `data/result_i.jsonl(.gz)` を順に読み込み
2. 各文献の `"title"`, `"abstract"`, `"description"`, `"claims"` を連結
3. 連結テキストを **文字数スライディング分割**（チャンク化）
4. 各チャンクを **API埋め込み → L2正規化**
5. 全チャンクをまとめて **FAISS IndexFlatIP（内積＝コサイン）** に登録
6. 出力を `rag_index/` に保存

---

### ⚙️ 主な引数

| 引数                | 意味                               |
| ----------------- | -------------------------------- |
| `--data_dir`      | JSONL入力フォルダ                      |
| `--out_dir`       | 出力先 (`rag_index`)                |
| `--select`        | 対象ファイル番号 `"1,3-5,12"`            |
| `--chunk_size`    | チャンク長（例: 1200文字）                 |
| `--chunk_overlap` | 重複（例: 200文字）                     |
| `--limit_docs`    | 最大文献数（0=全件）                      |
| `--provider`      | `gemini` または `openai_compat`     |
| `--emb_model`     | 使用モデル（例: `models/embedding-001`） |
| `--api_key_env`   | APIキーの環境変数名                      |

---

### 🧠 埋め込みロジック（内部処理）

#### (1) 文献の本文抽出と結合

```python
text = "\n\n".join([
    obj.get("title", ""),
    obj.get("abstract", ""),
    obj.get("description", ""),
    obj.get("claims", "")
]).strip()
```

#### (2) チャンク化（文字数ベース）

```python
def chunk_text(text, chunk_size=1200, overlap=200):
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text): break
        start = max(0, end - overlap)
    return chunks
```

→ 各文献は複数のチャンク（`#p0`, `#p1`, …）に分割。

#### (3) 埋め込み生成

```python
vec = client.embed(chunk_texts)  # Gemini or OpenAI互換API
vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)  # L2正規化
```

#### (4) FAISSへの登録

```python
index = faiss.IndexFlatIP(dim)  # 内積＝コサイン
index.add(vectors)
```

---

### 💾 出力ファイル

| ファイル                             | 内容                        |
| -------------------------------- | ------------------------- |
| `faiss.index`                    | 内積ベースのベクトル索引              |
| `vectors.npy`                    | チャンクごとの正規化ベクトル            |
| `docstore.jsonl`                 | `id`, `parent_id`, `text` |
| `emb_ids.npy` / `parent_ids.npy` | ID対応表                     |
| `manifest.json`                  | 実行条件の記録                   |

---

## 🚀 3. 検索＋評価（batch_search_score.py）

### 🧭 処理概要

1. クエリ側：出願文献 (`result_i.jsonl`) から
   `"title"`, `"abstract"`, `"description"`, `"claims"` を結合
2. クエリ本文を UTF-8安全に切り詰め＋分割 (`piece_chars`)
3. 各ピースを埋め込み → **平均ベクトル** を作成
4. **FAISSで全チャンクとの類似度（内積）検索**
5. 同一 `parent_id` のチャンクが複数ヒットした場合：
   → 最上位スコア1件のみ残す
6. 上位 mMax 件の親文献を評価に使用

---

### ⚙️ 検索ロジック

#### (1) クエリ埋め込み

```python
pieces = [text[i:i+piece_chars] for i in range(0, len(text), piece_chars)]
vecs = [client.embed_one(p) / ||p|| for p in pieces]
qv = np.mean(vecs, axis=0)
qv = qv / ||qv||
```

#### (2) 検索（類似度 = コサイン）

```python
D, I = faiss_index.search(qv.reshape(1, -1), k)
# D: 類似度スコア（内積）, I: チャンクindex
```

#### (3) チャンク→親文献に変換

```python
hits = [{"id": ids[i], "parent_id": parents[i], "score": D[0][rank]}
        for rank, i in enumerate(I[0])]
```

#### (4) 重複排除（親文献単位）

```python
seen = set()
filtered = []
for h in hits:
    if h["parent_id"] not in seen:
        seen.add(h["parent_id"])
        filtered.append(h)
top_results = filtered[:mMax]
```

→ 同じ出願の複数チャンクが上位に来た場合、**最高スコアの1チャンクのみ採用。**

---

### 🧪 評価指標

* Hit@k（正解文献が上位k件に存在）
* MRR（平均逆順位）
* Coverage（正解が存在するクエリ率）
* Precision@mMax（平均精度）

出力：

```
retrieved_pairs.csv
score_results/summary.csv
score_results/overall_summary.txt
```

---

## 🧾 4. 実行例

### インデックス構築（Gemini使用）

```bash
python build_index.py \
  --data_dir ./data \
  --out_dir ./rag_index \
  --provider gemini \
  --emb_model models/embedding-001 \
  --api_key_env GOOGLE_API_KEY \
  --select "1-3" \
  --chunk_size 1200 --chunk_overlap 200
```

### 一括検索＋評価

```bash
python batch_search_score.py \
  --data_dir ./data \
  --index_dir ./rag_index \
  --select "4" \
  --truth ./data/CSV1.csv ./data/CSV2.csv \
  --k 50 --mMax 10 --P 0.8 \
  --provider gemini \
  --emb_model models/embedding-001 \
  --api_key_env GOOGLE_API_KEY
```

---

## 🧮 5. 内部の数理的対応関係

| 項目            | 処理         | 数理的意味        |
| ------------- | ---------- | ------------ |
| L2正規化         | 各ベクトルを単位長に | コサイン距離＝内積    |
| IndexFlatIP   | 内積ベースFAISS | 余弦類似度検索      |
| チャンク分割        | 文書の局所特徴保持  | 長文内の部分意味単位   |
| 平均ベクトル        | 部分埋め込みの集約  | センテンスプーリング   |
| parent_id重複除去 | 1文献＝1スコア代表 | 出願単位のランキング評価 |

---

## ✅ 6. まとめ（処理全体の流れ）

```
[既存文献]
  └─ build_index.py
       ├─ title+abstract+description+claims を連結
       ├─ 文字単位チャンク化（overlap付き）
       ├─ 各チャンクを埋め込み → 正規化
       ├─ FAISS(IndexFlatIP) に登録
       └─ rag_index/ に保存

[クエリ文献]
  └─ batch_search_score.py
       ├─ 同様に連結＆分割 → 埋め込み平均化
       ├─ FAISSで全チャンクと類似度検索
       ├─ 同一parent_idは最上位のみ残す
       ├─ 上位mMax件で評価
       └─ summary.csv / overall_summary.txt 出力
```

---

この README は、
`build_index.py` と `batch_search_score.py` の **実際のコード仕様に100%一致した技術ドキュメント** です。
これ1本で埋め込み・チャンク分割・FAISS検索・評価の全挙動が再現可能です。
