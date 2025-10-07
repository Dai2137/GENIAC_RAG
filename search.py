# search.py
import faiss, json, numpy as np, argparse
from sentence_transformers import SentenceTransformer

INDEX_DIR = "./rag_index"
MODEL_NAME_FALLBACK = "intfloat/multilingual-e5-base"

def load_index():
    index = faiss.read_index(f"{INDEX_DIR}/index.faiss")
    ids = np.load(f"{INDEX_DIR}/emb_ids.npy", allow_pickle=True)
    parents = np.load(f"{INDEX_DIR}/parent_ids.npy", allow_pickle=True)
    try:
        meta = json.load(open(f"{INDEX_DIR}/meta.json","r",encoding="utf-8"))
        model_name = meta.get("model", MODEL_NAME_FALLBACK)
    except:
        meta, model_name = {}, MODEL_NAME_FALLBACK
    return index, ids.tolist(), parents.tolist(), meta, model_name

def encode_query(model, q: str):
    v = model.encode([f"query: {q}"], normalize_embeddings=True)
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        v = v.reshape(1,-1)
    return v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="query string")
    ap.add_argument("--k", type=int, default=10, help="top-k")
    ap.add_argument("--group_by_parent", action="store_true", help="親公開番号でまとめ直す")
    args = ap.parse_args()

    index, ids, parents, meta, model_name = load_index()
    model = SentenceTransformer(model_name)

    qv = encode_query(model, args.q)
    sims, idxs = index.search(qv, args.k)
    idxs, sims = idxs[0], sims[0]

    hits = []
    for rank, (i, s) in enumerate(zip(idxs, sims), 1):
        hits.append({
            "rank": rank,
            "chunk_id": ids[i],
            "parent_id": parents[i],
            "score": float(s),
        })

    if args.group_by_parent:
        # 同一親をまとめて最良スコアの順
        best = {}
        for h in hits:
            p = h["parent_id"]
            if p not in best or h["score"] > best[p]["score"]:
                best[p] = h
        hits = sorted(best.values(), key=lambda x: -x["score"])[:args.k]
        for i,h in enumerate(hits,1):
            h["rank"] = i

    print(json.dumps({"query": args.q, "results": hits}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
