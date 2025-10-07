#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSONL/XML → チャンク → 埋め込み → FAISS インデックス作成（ローカル・CPU最適化）
- data/ 配下の *.jsonl / *.jsonl.gz を自動検出
- JSONL: 1行=1文献(dict) を想定（あなたの変換ノートの出力形式）
- XML行にも一応対応（<doc-number> を ID に採用）
- E5-small を既定（速い/軽い）。max_seq_length/バッチ/CPUスレッド調整済
- 途中保存＆レジューム、重複排除、HNSW の近似検索
"""

import os, re, io, gzip, json, ujson, faiss, math, argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ---------------------------
# 引数
# ---------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default=str(Path(__file__).parent / "data"),
               help="JSONL(.gz)/XML行ファイルがあるディレクトリ")
ap.add_argument("--out_dir", default=str(Path(__file__).parent / "rag_index"),
               help="インデックス出力先")
ap.add_argument("--model", default="intfloat/multilingual-e5-small",
               help="SentenceTransformers の埋め込みモデル")
ap.add_argument("--batch", type=int, default=64, help="埋め込み時のバッチ")
ap.add_argument("--max_seq", type=int, default=256, help="埋め込み時の最大トークン長")
ap.add_argument("--chunk_size", type=int, default=1200, help="文字ベースのチャンク長")
ap.add_argument("--chunk_overlap", type=int, default=200, help="チャンクの重複長")
ap.add_argument("--resume", action="store_true", help="途中キャッシュからレジューム")
ap.add_argument("--hnsw", action="store_true", help="HNSW 近似検索を使う（デフォルト: Flat）")
ap.add_argument("--limit_docs", type=int, default=0, help="先頭から文書数を制限(0で無制限)")
ap.add_argument("--threads", type=int, default=max(1, os.cpu_count() // 2), help="CPUスレッド数")
args = ap.parse_args()

DATA_DIR = Path(args.data_dir).resolve()
OUT_DIR  = Path(args.out_dir).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CPU最適化（GPUは使わない前提）
import torch
torch.set_num_threads(args.threads)
os.environ["OMP_NUM_THREADS"] = str(args.threads)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# ---------------------------
# 1) 入力列挙
# ---------------------------
def list_inputs():
    files = []
    # まず明示的に *.jsonl / *.jsonl.gz
    files.extend(sorted(DATA_DIR.glob("*.jsonl")))
    files.extend(sorted(DATA_DIR.glob("*.jsonl.gz")))
    # 念のため *.txt（XML行）も対象に
    files.extend(sorted(DATA_DIR.glob("*.txt")))
    return [p for p in files if p.is_file()]

INPUT_FILES = list_inputs()
if not INPUT_FILES:
    raise SystemExit(f"[ERROR] 入力が見つかりません: {DATA_DIR}")

print(f"[INFO] 検出ファイル数: {len(INPUT_FILES)}")
for p in INPUT_FILES[:5]:
    print("  -", p.name)
if len(INPUT_FILES) > 5:
    print("  ...")

# ---------------------------
# 2) テキスト抽出（JSONL 1行=1文献 or XML行）
# ---------------------------
ID_CANDIDATES = ["publication_number","doc_number","publication_id","pub_id","jp_pub_no","id"]
TEXT_FIELDS   = ["title","abstract","description","claims","text","body","sections","paragraphs","xml"]
DOCNUM_RE     = re.compile(r"<doc-number>\s*([0-9A-Za-z\-]+)\s*</doc-number>", re.I)

def flatten_text(x):
    out=[]
    def rec(v):
        if v is None: return
        if isinstance(v, str):
            s=v.strip()
            if s: out.append(s)
        elif isinstance(v, (list, tuple)):
            for t in v: rec(t)
        elif isinstance(v, dict):
            for t in v.values(): rec(t)
    rec(x)
    return re.sub(r"\s+"," ", " ".join(out)).strip()

def parse_json_line(line: str):
    obj = ujson.loads(line)
    pid = None
    for k in ID_CANDIDATES:
        if k in obj:
            pid = str(obj[k]); break
    if not pid and isinstance(obj.get("publication"), dict):
        for k in ID_CANDIDATES:
            if k in obj["publication"]:
                pid = str(obj["publication"][k]); break
    parts=[]
    for k in TEXT_FIELDS:
        if k in obj:
            t = flatten_text(obj[k])
            if t: parts.append(t)
    if not parts:
        # JSON内の文字列にXMLが入っているケース
        for v in obj.values():
            if isinstance(v,str) and "<jp-official-gazette" in v:
                m = DOCNUM_RE.search(v)
                pid = pid or (m.group(1) if m else None)
                parts.append(re.sub(r"\s+"," ", v).strip())
                break
    if not parts:
        return None
    if not pid:
        # 最後の手段：ハッシュ
        pid = f"ANON_{hash(' '.join(parts))%10**12}"
    return {"id": pid, "text": " \n".join(parts)}

def parse_xml_line(line: str):
    if "<jp-official-gazette" not in line and "<?xml" not in line:
        return None
    m = DOCNUM_RE.search(line)
    pid = m.group(1) if m else None
    if not pid:
        pid = f"ANON_{hash(line)%10**12}"
    return {"id": pid, "text": re.sub(r"\s+"," ", line).strip()}

def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")

def stream_docs(paths):
    n = 0
    for p in paths:
        with open_text(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    if line[:1] in "{[":
                        doc = parse_json_line(line)
                    else:
                        doc = parse_xml_line(line)
                    if doc and doc["text"]:
                        yield doc
                        n += 1
                        if args.limit_docs and n >= args.limit_docs:
                            return
                except Exception:
                    continue

# ---------------------------
# 3) チャンク分割 & 重複排除
# ---------------------------
def chunk_text(t: str, size: int, overlap: int):
    t = t.strip()
    if len(t) <= size:
        return [t]
    chunks = []
    start = 0
    while start < len(t):
        end = min(len(t), start + size)
        chunks.append(t[start:end])
        if end == len(t): break
        start = max(end - overlap, start + 1)
    return chunks

def uniq_by_hash(ids, texts, parents):
    seen = set(); ui=[]; ut=[]; up=[]
    for i,t,p in zip(ids,texts,parents):
        h = hash(t)
        if h in seen: continue
        seen.add(h); ui.append(i); ut.append(t); up.append(p)
    return ui, ut, up

# ---------------------------
# 4) 埋め込み
# ---------------------------
from sentence_transformers import SentenceTransformer

print(f"[INFO] Embedding model: {args.model}")
model = SentenceTransformer(args.model, device="cpu")
try:
    model.max_seq_length = args.max_seq
except Exception:
    pass

def encode_passages(texts):
    texts = [f"passage: {t}" for t in texts]
    all_vecs=[]
    with torch.inference_mode():
        for i in tqdm(range(0, len(texts), args.batch), desc="Encoding", mininterval=0.2):
            batch = texts[i:i+args.batch]
            emb = model.encode(batch, normalize_embeddings=True,
                               batch_size=args.batch, convert_to_tensor=True,
                               show_progress_bar=False)
            emb = emb.detach().cpu().numpy().astype("float32")
            all_vecs.append(emb)
    return np.vstack(all_vecs)

# ---------------------------
# メイン：スキャン→チャンク→埋め込み→FAISS保存
# ---------------------------
print("[INFO] Scanning & chunking...")
ids, texts, parents = [], [], []
for doc in tqdm(stream_docs(INPUT_FILES), desc="Reading"):
    pid = str(doc["id"])
    for i, ch in enumerate(chunk_text(doc["text"], args.chunk_size, args.chunk_overlap)):
        ids.append(f"{pid}#p{i}")
        texts.append(ch)
        parents.append(pid)

if not ids:
    raise SystemExit("[ERROR] 文書0件。入力の形式をご確認ください。")

ids, texts, parents = uniq_by_hash(ids, texts, parents)
print(f"[INFO] chunks: {len(ids)} (after dedup)")

# 埋め込み
emb = encode_passages(texts)
dim = emb.shape[1]
print("[INFO] Embeddings:", emb.shape)

# FAISS index
if args.hnsw:
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200
    index.add(emb)
    index.hnsw.efSearch = 64
else:
    index = faiss.IndexFlatIP(dim)  # 正規化済み→cos類似=内積
    index.add(emb)

# 保存
faiss.write_index(index, str(OUT_DIR / "index.faiss"))
np.save(OUT_DIR / "emb_ids.npy", np.array(ids, dtype=object))
np.save(OUT_DIR / "parent_ids.npy", np.array(parents, dtype=object))
with open(OUT_DIR / "chunks.jsonl", "w", encoding="utf-8") as wf:
    for i, t in enumerate(texts):
        wf.write(ujson.dumps({"id": ids[i], "parent": parents[i], "text": t}, ensure_ascii=False) + "\n")
with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
    json.dump({
        "model": args.model,
        "batch": args.batch,
        "max_seq_length": args.max_seq,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "hnsw": args.hnsw,
        "count_chunks": len(ids),
        "threads": args.threads,
    }, f, ensure_ascii=False, indent=2)

print("[DONE] Saved to:", OUT_DIR)
