#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 使い方（例）
# OpenAI互換API（国内ベンダ想定）
# python search.py \
#   --index_dir ./rag_index \
#   --provider openai_compat \
#   --api_base https://api.example.com/v1 \
#   --emb_model embedding-japanese-v1 \
#   --api_key_env EMB_API_KEY \
#   --q "リチウムイオン電池の電極材料" \
#   --k 20 \
#   --group_by_parent

# Gemini Embedding API
# python search.py \
#   --index_dir ./rag_index \
#   --provider gemini \
#   --emb_model models/embedding-001 \
#   --api_key_env GOOGLE_API_KEY \
#   --q "AIによる画像認識アルゴリズム" \
#   --k 20 \
#   --group_by_parent



import os, json, argparse, time, requests, sys
import numpy as np
import faiss
from pathlib import Path
from typing import List

# ===================== 引数 =====================
ap = argparse.ArgumentParser()
ap.add_argument("--index_dir", required=True, help="build_index.py で作成したインデックスディレクトリ")
ap.add_argument("--q", required=True, help="検索クエリ")
ap.add_argument("--k", type=int, default=10, help="取得件数（親文献に集約後もこの件数に揃える）")
ap.add_argument("--group_by_parent", action="store_true", help="親文献単位に集約（推奨）")

# Embedding API（build_index と同じ指定が可能）
ap.add_argument("--provider", choices=["openai_compat", "gemini"], default=None,
                help="meta.json があれば自動補完。上書きしたいときに指定")
ap.add_argument("--api_base", default="", help="OpenAI互換APIのベースURL (例: https://api.xxx/v1)")
ap.add_argument("--api_key_env", default=None, help="APIキーを入れた環境変数名（例: EMB_API_KEY / GOOGLE_API_KEY）")
ap.add_argument("--emb_model", default=None, help="Embeddingモデル名（例: embedding-japanese-v1 / models/embedding-001）")
ap.add_argument("--rpm", type=int, default=600, help="リクエスト毎分上限（クエリは少数なので大きめでOK）")
args = ap.parse_args()

IDX = Path(args.index_dir)
index = faiss.read_index(str(IDX / "index.faiss"))
ids = np.load(IDX / "emb_ids.npy", allow_pickle=True).tolist()
parents = np.load(IDX / "parent_ids.npy", allow_pickle=True).tolist()

# メタ情報（build_index.py が出力）
meta_path = IDX / "meta.json"
meta = {}
if meta_path.exists():
    try:
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
    except Exception:
        meta = {}

# 優先順位：CLI引数 > meta.json > 既定値
provider = args.provider or meta.get("provider") or "openai_compat"
emb_model = args.emb_model or meta.get("emb_model") or "embedding-japanese-v1"
api_base = args.api_base or meta.get("api_base") or ""
api_key_env = args.api_key_env or ("GOOGLE_API_KEY" if provider=="gemini" else "EMB_API_KEY")
api_key = os.environ.get(api_key_env, "")

if provider=="openai_compat" and not api_base:
    raise SystemExit("[ERROR] provider=openai_compat の場合 --api_base を指定してください（例: https://api.example.com/v1）")
if not api_key:
    raise SystemExit(f"[ERROR] APIキーが未設定です。環境変数 {api_key_env} をセットしてください。")

# =============== Embedding クライアント ===============
class EmbeddingClient:
    def __init__(self, provider:str, api_base:str, model:str, api_key:str, rpm:int):
        self.provider = provider
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.interval = max(0.001, 60.0 / max(1, rpm))
        self.last_call = 0.0

        if provider=="gemini":
            try:
                import google.generativeai as genai  # noqa: F401
            except Exception:
                print("[ERROR] provider=gemini には 'google-generativeai' が必要です: pip install google-generativeai", file=sys.stderr)
                raise

    def _rate(self):
        now=time.time()
        wait=self.last_call + self.interval - now
        if wait>0: time.sleep(wait)
        self.last_call = time.time()

    def embed_query(self, text:str) -> np.ndarray:
        self._rate()
        if self.provider=="openai_compat":
            return self._embed_openai_compat(text)
        elif self.provider=="gemini":
            return self._embed_gemini(text)
        else:
            raise ValueError(f"unknown provider: {self.provider}")

    def _embed_openai_compat(self, text:str) -> np.ndarray:
        url = f"{self.api_base}/embeddings"
        headers = {"Content-Type":"application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "input": [text]}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"[OpenAI-compat] {r.status_code}: {r.text[:200]}")
        data = r.json()
        v = np.array(data["data"][0]["embedding"], dtype=np.float32)
        # FAISS IndexFlatIP 前提なので単位ベクトルに正規化
        v = v / (np.linalg.norm(v) + 1e-12)
        return v.astype("float32")[None, :]

    def _embed_gemini(self, text:str) -> np.ndarray:
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        res = genai.embed_content(model=self.model, content=text)
        v = np.array(res["embedding"], dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-12)
        return v.astype("float32")[None, :]

client = EmbeddingClient(provider, api_base, emb_model, api_key, rpm=args.rpm)

# ======== クエリ前処理（必要ならプロンプト語を付ける）========
# 多くのpassage系モデルは "query: " を付ける設計が一般的だが、
# OpenAI互換APIのembeddingモデルは不要なことが多い。metaのproviderに合わせて切替。
def format_query(q: str) -> str:
    if "e5" in (emb_model or "").lower():
        return f"query: {q}"
    # それ以外は素のまま
    return q

qtext = format_query(args.q)
qv = client.embed_query(qtext)  # (1, dim)

# =============== 検索 ===============
sims, idxs = index.search(qv, args.k if not args.group_by_parent else min(args.k*5, len(ids)))
sims, idxs = sims[0], idxs[0]

hits = []
for rank, (i, s) in enumerate(zip(idxs, sims), 1):
    hits.append({"rank": rank, "id": ids[i], "parent_id": parents[i], "score": float(s)})

# 親文献で集約（親ごとに最高スコア）
if args.group_by_parent:
    best = {}
    first_order = {}
    for h in hits:
        p = h["parent_id"]
        if (p not in best) or (h["score"] > best[p]["score"]):
            best[p] = h
            first_order.setdefault(p, h["rank"])
    hits = sorted(best.values(), key=lambda x: (-x["score"], first_order[x["parent_id"]]))[:args.k]
    for i,h in enumerate(hits,1): h["rank"]=i

print(json.dumps({"query": args.q, "provider": provider, "model": emb_model, "results": hits}, ensure_ascii=False, indent=2))
