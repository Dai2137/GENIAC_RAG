# GENIAC-PRIZE Patent Retrieval (RAG Pipeline)

特許文献データを用いて、
**埋め込み（国内/汎用API） → ベクトル索引（FAISS） → 検索 → 評価**
を行う最小構成のプロジェクトです。GPU不要・再現性重視。

* 埋め込みは **国内ベンダの OpenAI 互換API**（ELYZA / rinna / ABEJA 等）または **Gemini** を利用
* ベクトル検索は **FAISS（CPU）**
* GENIAC（官公庁コンペ）を念頭に、**事前計算→当日安定稼働**の設計

---

## 0. 前提（これだけあればOK）

* **Python 3.10+**（3.11 推奨）
* OS: macOS / Linux / Windows（PowerShell）
* 取得済みの特許データ（`*.jsonl` or `*.jsonl.gz` or 行ごとXML文字列）

  * 例: `data/result_1/` 配下に `result_1.jsonl` など
* いずれかの **Embedding API キー**

  * 国内API（OpenAI互換エンドポイント） → 例: `EMB_API_KEY`
  * Google Gemini → `GOOGLE_API_KEY`

---

## 1. ディレクトリ構成

```text
your-project/
├─ build_index.py         # 埋め込み作成 & FAISS索引作成
├─ search.py              # クエリ埋め込み → 検索
├─ score_explore.py       # GENIAC公式CSVに基づく評価＋ファイル出力
├─ requirements.txt
├─ data/
│   ├─ result_1.jsonl     # 特許文献（入力データ）
│   ├─ CSV1.csv           # 公式AX/AY対応CSV（syutugan, category, himotuki）
│   └─ CSV2.csv
├─ rag_index/             # 索引出力（自動生成）
│   ├─ index.faiss
│   ├─ emb_ids.npy
│   ├─ parent_ids.npy
│   ├─ chunks.jsonl
│   └─ meta.json
├─ search_result.json     # 検索結果（出力）
└─ score_results/         # 評価結果（自動保存）
    └─ JP2012239158A_result.json

```

---

## 2. 環境構築（仮想環境＋依存の導入）

### macOS / Linux（Bash）

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Windows（PowerShell）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

**requirements.txt（最小）**

```txt
faiss-cpu
numpy
pandas
ujson
tqdm
requests
google-generativeai    # Geminiを使う場合に必要
```

> GPUは不要です。もし自前GPUで使いたい人は `faiss-gpu` に置き換え可。

---

## 3. APIキーの設定

### 国内API（OpenAI互換）を使う場合

```bash
# macOS / Linux
export EMB_API_KEY="あなたのAPIキー"
# Windows PowerShell
# $env:EMB_API_KEY="あなたのAPIキー"
```

> **エンドポイント（api_base）例**: `https://api.example.com/v1`
> **モデル名（emb_model）例**: `embedding-japanese-v1`
> ベンダの仕様書の値をそのまま指定してください。

### Geminiを使う場合

```bash
# macOS / Linux
export GOOGLE_API_KEY="あなたのAPIキー"
# Windows PowerShell
# $env:GOOGLE_API_KEY="あなたのAPIキー"
```

---

## 4. データを置く（入力形式）

* JSONL（1行＝1レコード）を推奨。XML文字列1行でも可。
* 下記のいずれかのキーがあればIDとして優先取得：
  `publication_number, doc_number, publication_id, pub_id, jp_pub_no, id`
* テキストは下記フィールドを結合：
  `title, abstract, description, claims, text, body, sections, paragraphs, xml`

> 例：`data/result_1/result_1.jsonl` を用意。

---

## 5. 埋め込み作成 & インデックス構築（build_index.py）

### 国内API（OpenAI互換）

```bash
python build_index.py \
  --data_dir ./data \
  --out_dir  ./rag_index \
  --provider openai_compat \
  --api_base https://api.example.com/v1 \
  --emb_model embedding-japanese-v1 \
  --api_key_env EMB_API_KEY \
  --batch 128 --rpm 180 \
  --chunk_size 1200 --chunk_overlap 200 \
  --limit_docs 10
```

### Gemini

```bash
python build_index.py \
  --data_dir ./data \
  --out_dir  ./rag_index \
  --provider gemini \
  --emb_model models/embedding-001 \
  --api_key_env GOOGLE_API_KEY \
  --batch 32 --rpm 300 \
  --chunk_size 1200 --chunk_overlap 200 \
  --limit_docs 10
```

**出力（自動生成）**

```
rag_index/
 ├─ index.faiss        # ベクトル索引
 ├─ emb_ids.npy        # チャンクID
 ├─ parent_ids.npy     # 親文献ID（公開番号など）
 ├─ chunks.jsonl       # チャンクの中身（デバッグ用）
 └─ meta.json          # 実行時メタ（再検索に再利用）
```

> 既定は **FAISS IndexFlatIP（厳密検索）× 正規化ベクトル**。
> 大規模化時は量子化/圧縮は別途（本READMEでは安全優先）。

---

## 6. 検索（search.py）

* まずは**親文献単位の集約**を有効化するのが実用的（`--group_by_parent`）
* 使う埋め込みAPIは**インデックス作成時の設定に合わせる**

### 国内API（OpenAI互換）

```bash
python search.py \
  --index_dir ./rag_index \
  --provider openai_compat \
  --api_base https://api.example.com/v1 \
  --emb_model embedding-japanese-v1 \
  --api_key_env EMB_API_KEY \
  --q "カーボンナノチューブの製造方法" \
  --k 20 \
  --group_by_parent > search_result.json
```

### Gemini

```bash
python search.py \
  --index_dir ./rag_index \
  --provider gemini \
  --emb_model models/embedding-001 \
  --api_key_env GOOGLE_API_KEY \
  --q "有機ELディスプレイの封止構造" \
  --k 20 \
  --group_by_parent > search_result.json
```

**出力例（`search_result.json`）**

```json
{
  "query": "カーボンナノチューブの製造方法",
  "provider": "openai_compat",
  "model": "embedding-japanese-v1",
  "results": [
    {"rank": 1, "id": "JP2020123456A#p0", "parent_id": "JP2020123456A", "score": 0.882},
    {"rank": 2, "id": "JP2019009876A#p2", "parent_id": "JP2019009876A", "score": 0.851}
  ]
}
```

---

## 7. 評価（score_explore.py：GENIACのAX/AY仕様）
✅ 配布CSVの列構造
| 列名         | 意味         | 備考                |
| ---------- | ---------- | ----------------- |
| `syutugan` | 対象出願（評価対象） | 公開番号              |
| `category` | 区分         | `"Ax"` または `"Ay"` |
| `himotuki` | 紐付き文献      | 参照特許の公開番号         |
| `koukaibi` | 効果日（任意）    | スコア計算には非使用        |

**実行コマンド**
```bash
python score_explore.py \
  --truth ./data/CSV1.csv ./data/CSV2.csv \
  --syutugan JP2012239158A \
  --retrieved_json ./search_result.json \
  --k 50 --mMax 10 --P 0.8
```

**出力例**

```
score_results/
 ├─ JP2012239158A_result.json
 ├─ JP2012239158A_result.json
 ├─ JP2012239158A_result.json
 ├─ JP2012239158A_result.json
 ├─ JP2012239158A_result.json
 └─ ...
```


```json
{
  "syutugan": "JP2012239158A",
  "Nax": 1,
  "Nay": 4,
  "n": 5,
  "m": 20,
  "mMin": 7,
  "mMax": 10,
  "P": 0.8,
  "score_scaled": 93.3,
  "ax_hit": true,
  "ay_hit": 2
}

```

---

## 8. よくある質問（FAQ）

**Q1. `faiss-cpu` は必須？**
A. 必須です（インデックス作成・検索に使用）。

**Q2. 429/5xx が出る／途中で落ちる**
A. `build_index.py` は指数バックオフ＋レート制御済みです。`--rpm` と `--batch` を下げて再実行。途中成果は `rag_index/` に逐次保存されます。

**Q3. モデル次元は自動で合わせてる？**
A. はい。APIの埋め込み次元に合わせてFAISSを作成し、クエリ側も正規化してから検索します。

**Q4. クエリの書き方のコツは？**
A. 技術課題・発明主題で表現するのが最も効果的（例：「ドローンの姿勢制御装置」「ポリ乳酸系樹脂の改質方法」）。

**Q5. データが巨大で時間がかかる**
A. `--limit_docs` でサンプルから始め、結線が正しいことを確認してから全量に拡張してください。

---

## 9. 代表的コマンド（コピペ用）

**国内API一気通貫**

```bash
# 1) 環境
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
export EMB_API_KEY="YOUR_KEY"

# 2) 埋め込み→索引
python build_index.py \
  --data_dir ./data \
  --out_dir  ./rag_index \
  --provider openai_compat \
  --api_base https://api.example.com/v1 \
  --emb_model embedding-japanese-v1 \
  --api_key_env EMB_API_KEY

# 3) 検索
python search.py \
  --index_dir ./rag_index \
  --provider openai_compat \
  --api_base https://api.example.com/v1 \
  --emb_model embedding-japanese-v1 \
  --api_key_env EMB_API_KEY \
  --q "ドローンの姿勢制御装置" \
  --k 20 --group_by_parent > search_result.json

# 4) 評価（任意）
python score_explore.py \
  --truth ./ax_ay_truth.csv \
  --alpha JP2020123456A \
  --retrieved_json ./search_result.json
```

**Gemini一気通貫**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
export GOOGLE_API_KEY="YOUR_KEY"

python build_index.py \
  --data_dir ./data \
  --out_dir  ./rag_index \
  --provider gemini \
  --emb_model models/embedding-001 \
  --api_key_env GOOGLE_API_KEY

python search.py \
  --index_dir ./rag_index \
  --provider gemini \
  --emb_model models/embedding-001 \
  --api_key_env GOOGLE_API_KEY \
  --q "有機ELディスプレイの封止構造" \
  --k 20 --group_by_parent > search_result.json
```

---

## 10. 変更履歴（このREADME対応）

* SentenceTransformer依存を排除し、**APIベース埋め込み**に統一
* `build_index.py` / `search.py` / `score_explore.py` を**国内API・Gemini両対応**
* **親文献集約・重複排除・上位K**の安定ロジック実装
* **GENIACのAX/AY評価**をそのまま流せるCLI
* 評価結果は常に score_results/ に保存（ターミナルにも表示）