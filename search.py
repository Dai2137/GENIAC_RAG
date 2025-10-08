#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, argparse, numpy as np, faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--index_dir", default="/content/rag_index")
ap.add_argument("--model", default=None)
ap.add_argument("--q", required=True)
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--group_by_parent", action="store_true")
args = ap.parse_args([])  # Colabセルで直接実行するなら []、スクリプトなら削除

IDX = Path(args.index_dir)
index = faiss.read_index(str(IDX / "index.faiss"))
ids = np.load(IDX / "emb_ids.npy", allow_pickle=True).tolist()
parents = np.load(IDX / "parent_ids.npy", allow_pickle=True).tolist()

try:
    meta = json.load(open(IDX / "meta.json","r",encoding="utf-8"))
    model_name = meta.get("model") or args.model or "intfloat/multilingual-e5-small"
except Exception:
    meta = {}
    model_name = args.model or "intfloat/multilingual-e5-small"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(model_name, device=device)
if "max_seq_length" in meta: model.max_seq_length = meta["max_seq_length"]

def encode_query(q: str):
    v = model.encode([f"query: {q}"], normalize_embeddings=True,
                     batch_size=1, convert_to_tensor=True, show_progress_bar=False)
    return v.detach().float().cpu().numpy().astype("float32")

qv = encode_query(args.q)
sims, idxs = index.search(qv, args.k)
idxs, sims = idxs[0], sims[0]

hits = []
for rank, (i, s) in enumerate(zip(idxs, sims), 1):
    hits.append({"rank": rank, "chunk_id": ids[i], "parent_id": parents[i], "score": float(s)})

if args.group_by_parent:
    best = {}
    for h in hits:
        p = h["parent_id"]
        if p not in best or h["score"] > best[p]["score"]:
            best[p] = h
    hits = sorted(best.values(), key=lambda x: -x["score"])[:args.k]
    for i,h in enumerate(hits,1): h["rank"]=i

print(json.dumps({"query": args.q, "results": hits}, ensure_ascii=False, indent=2))
