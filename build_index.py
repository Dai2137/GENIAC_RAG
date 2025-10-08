#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, io, gzip, json, ujson, faiss, argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default="/content/drive/MyDrive/Colab Notebooks/GENIAC", help="*.jsonl(.gz) を置いた場所")
ap.add_argument("--out_dir",  default="/content/rag_index", help="出力先")
ap.add_argument("--model",    default="intfloat/multilingual-e5-small")
ap.add_argument("--batch",    type=int, default=256)      # GPU なので大きめ
ap.add_argument("--max_seq",  type=int, default=256)
ap.add_argument("--chunk_size", type=int, default=1200)
ap.add_argument("--chunk_overlap", type=int, default=200)
ap.add_argument("--use_gpu_faiss", action="store_true", help="FAISSをGPUに載せる(IndexFlatIPのみ)")
ap.add_argument("--limit_docs", type=int, default=0)
args = ap.parse_args([])  # ← Colabノートから直接実行する想定。スクリプト実行なら [] を削除

DATA_DIR = Path(args.data_dir)
OUT_DIR  = Path(args.out_dir); OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 入力列挙 ----------
def list_inputs():
    files = []
    files += sorted(DATA_DIR.glob("*.jsonl"))
    files += sorted(DATA_DIR.glob("*.jsonl.gz"))
    files += sorted(DATA_DIR.glob("*.txt"))
    return [p for p in files if p.is_file()]

INPUT_FILES = list_inputs()
if not INPUT_FILES:
    raise SystemExit(f"[ERROR] no input under {DATA_DIR}")
print("[INFO] files:", len(INPUT_FILES))

# ---------- 1行=1文献 or XML行 パーサ ----------
ID_CANDIDATES = ["publication_number","doc_number","publication_id","pub_id","jp_pub_no","id"]
TEXT_FIELDS   = ["title","abstract","description","claims","text","body","sections","paragraphs","xml"]
DOCNUM_RE     = re.compile(r"<doc-number>\s*([0-9A-Za-z\-]+)\s*</doc-number>", re.I)

def flatten_text(x):
    out=[]
    def rec(v):
        if v is None: return
        if isinstance(v,str):
            s=v.strip()
            if s: out.append(s)
        elif isinstance(v,(list,tuple)):
            for t in v: rec(t)
        elif isinstance(v,dict):
            for t in v.values(): rec(t)
    rec(x)
    return re.sub(r"\s+"," ", " ".join(out)).strip()

def parse_json_line(line: str):
    obj = ujson.loads(line)
    pid = None
    for k in ID_CANDIDATES:
        if k in obj: pid = str(obj[k]); break
    if not pid and isinstance(obj.get("publication"), dict):
        for k in ID_CANDIDATES:
            if k in obj["publication"]: pid = str(obj["publication"][k]); break
    parts=[]
    for k in TEXT_FIELDS:
        if k in obj:
            t = flatten_text(obj[k])
            if t: parts.append(t)
    if not parts:
        for v in obj.values():
            if isinstance(v,str) and "<jp-official-gazette" in v:
                m = DOCNUM_RE.search(v)
                pid = pid or (m.group(1) if m else None)
                parts.append(re.sub(r"\s+"," ", v).strip())
                break
    if not parts: return None
    if not pid: pid = f"ANON_{hash(' '.join(parts))%10**12}"
    return {"id": pid, "text": " \n".join(parts)}

def parse_xml_line(line: str):
    if "<jp-official-gazette" not in line and "<?xml" not in line: return None
    m = DOCNUM_RE.search(line); pid = m.group(1) if m else f"ANON_{hash(line)%10**12}"
    return {"id": pid, "text": re.sub(r"\s+"," ", line).strip()}

def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="ignore") if path.suffix==".gz" else open(path, "r", encoding="utf-8", errors="ignore")

def stream_docs(paths):
    n=0
    for p in paths:
        with open_text(p) as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try:
                    doc = parse_json_line(line) if line[:1] in "{[" else parse_xml_line(line)
                    if doc and doc["text"]:
                        yield doc
                        n+=1
                        if args.limit_docs and n>=args.limit_docs: return
                except Exception:
                    continue

def chunk_text(t, size, overlap):
    t=t.strip()
    if len(t)<=size: return [t]
    out=[]; s=0
    while s<len(t):
        e=min(len(t), s+size)
        out.append(t[s:e])
        if e==len(t): break
        s=max(e-overlap, s+1)
    return out

def uniq_by_hash(ids, texts, parents):
    seen=set(); ui=[]; ut=[]; up=[]
    for i,t,p in zip(ids,texts,parents):
        h=hash(t)
        if h in seen: continue
        seen.add(h); ui.append(i); ut.append(t); up.append(p)
    return ui,ut,up

# ---------- 埋め込み（GPU/FP16） ----------
import torch
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] device:", device)

model = SentenceTransformer(args.model, device=device)
try: model.max_seq_length = args.max_seq
except: pass

def encode_passages(texts):
    texts = [f"passage: {t}" for t in texts]
    all_vecs=[]
    with torch.inference_mode():
        use_fp16 = (device=="cuda")
        ctx = torch.cuda.amp.autocast(dtype=torch.float16) if use_fp16 else nullcontext()
        with ctx:
            for i in tqdm(range(0, len(texts), args.batch), desc="Encoding"):
                batch = texts[i:i+args.batch]
                emb = model.encode(batch, normalize_embeddings=True,
                                   batch_size=args.batch, convert_to_tensor=True,
                                   show_progress_bar=False)
                emb = emb.detach().float().cpu().numpy().astype("float32")
                all_vecs.append(emb)
    return np.vstack(all_vecs)

from contextlib import nullcontext

# ---------- スキャン → チャンク ----------
ids,texts,parents=[],[],[]
for doc in tqdm(stream_docs(INPUT_FILES), desc="Reading"):
    pid=str(doc["id"])
    for j,ch in enumerate(chunk_text(doc["text"], args.chunk_size, args.chunk_overlap)):
        ids.append(f"{pid}#p{j}"); texts.append(ch); parents.append(pid)

if not ids: raise SystemExit("[ERROR] no chunks")

ids,texts,parents = uniq_by_hash(ids,texts,parents)
print("[INFO] chunks:", len(ids))

# ---------- エンコード ----------
emb = encode_passages(texts)
dim = emb.shape[1]
print("[INFO] embeddings:", emb.shape)

# ---------- FAISS（GPU/CPU） ----------
if args.use_gpu_faiss and device=="cuda":
    res = faiss.StandardGpuResources()
    index_cpu = faiss.IndexFlatIP(dim)   # 正確検索
    index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    index.add(emb)
else:
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

# ---------- 保存 ----------
faiss.write_index(faiss.index_gpu_to_cpu(index) if hasattr(index, "this") and isinstance(index, faiss.GpuIndexFlat) else index,
                  str(OUT_DIR / "index.faiss"))
np.save(OUT_DIR / "emb_ids.npy", np.array(ids, dtype=object))
np.save(OUT_DIR / "parent_ids.npy", np.array(parents, dtype=object))
with open(OUT_DIR / "chunks.jsonl","w",encoding="utf-8") as wf:
    for i,t in enumerate(texts):
        wf.write(ujson.dumps({"id": ids[i], "parent": parents[i], "text": t}, ensure_ascii=False)+"\n")
with open(OUT_DIR / "meta.json","w",encoding="utf-8") as f:
    json.dump({
        "model": args.model,
        "batch": args.batch,
        "max_seq_length": args.max_seq,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "device": device,
        "faiss_gpu": bool(args.use_gpu_faiss and device=="cuda"),
        "count_chunks": len(ids)
    }, f, ensure_ascii=False, indent=2)

print("[DONE] saved to:", OUT_DIR)
