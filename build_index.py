#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, io, gzip, json, ujson, faiss, argparse, time, math, sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
from contextlib import nullcontext
import requests
from typing import List, Dict, Any, Tuple

# ===================== 引数 =====================
ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default="./data", help="*.jsonl(.gz)/.txt を置いた場所")
ap.add_argument("--out_dir",  default="./rag_index", help="出力先")
# Embedding API 設定
ap.add_argument("--provider", choices=["openai_compat", "gemini"], default="openai_compat",
                help="国内APIは多くがOpenAI互換。Geminiも選択可")
ap.add_argument("--api_base", default="", help="OpenAI互換APIのベースURL (例: https://api.xxx/v1)")
ap.add_argument("--api_key_env", default="EMB_API_KEY", help="APIキーを格納した環境変数名")
ap.add_argument("--emb_model", default="embedding-japanese-v1", help="Embeddingモデル名")
ap.add_argument("--rpm", type=int, default=180, help="毎分リクエスト上限（レート制御）")
ap.add_argument("--batch", type=int, default=128, help="APIへの1回の入力件数")
ap.add_argument("--max_tokens_per_item", type=int, default=1200, help="1件の想定トークン上限（rate制御の目安）")
# 前処理・インデックス
ap.add_argument("--max_seq",  type=int, default=256, help="(互換のため残置; API側では無視されることが多い)")
ap.add_argument("--chunk_size", type=int, default=1200)
ap.add_argument("--chunk_overlap", type=int, default=200)
ap.add_argument("--use_gpu_faiss", action="store_true", help="FAISSをGPUに載せる(IndexFlatIPのみ)")
ap.add_argument("--limit_docs", type=int, default=0)
args = ap.parse_args()

DATA_DIR = Path(args.data_dir)
OUT_DIR  = Path(args.out_dir); OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===================== 入力列挙 =====================
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

# ===================== パース =====================
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

# ===================== 埋め込み API クライアント =====================
class EmbeddingClient:
    def __init__(self, provider:str, api_base:str, model:str, api_key:str, rpm:int, max_tokens_per_item:int):
        self.provider = provider
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.rpm = max(1, rpm)
        self.interval = 60.0 / self.rpm  # リクエスト間隔（秒）
        self.last_call = 0.0
        self.max_tokens_per_item = max_tokens_per_item

        if provider=="openai_compat" and not self.api_base:
            raise ValueError("--api_base は必須（例: https://api.example.com/v1）")
        if provider=="gemini":
            # pip install google-generativeai が必要
            try:
                import google.generativeai as genai  # noqa: F401
            except Exception:
                print("[ERROR] provider=gemini には 'google-generativeai' が必要です: pip install google-generativeai", file=sys.stderr)
                raise

    def _rate_limit(self):
        now = time.time()
        wait = self.last_call + self.interval - now
        if wait > 0:
            time.sleep(wait)
        self.last_call = time.time()

    def embed_batch(self, texts:List[str]) -> np.ndarray:
        # 長文のトークン超過はベンダ側で弾かれるので、ここではバイト長で軽いガード
        clipped = [t if len(t)<=20000 else t[:20000] for t in texts]  # 念のため20k文字でクリップ

        for attempt in range(6):  # 最大リトライ
            try:
                self._rate_limit()
                if self.provider == "openai_compat":
                    return self._embed_openai_compat(clipped)
                elif self.provider == "gemini":
                    return self._embed_gemini(clipped)
                else:
                    raise ValueError(f"unknown provider: {self.provider}")
            except Exception as e:
                # 429/5xx対策の指数バックオフ
                backoff = min(60.0, 2.0 ** attempt)
                print(f"[WARN] embed_batch retry {attempt+1}: {e}; sleep {backoff:.1f}s")
                time.sleep(backoff)
        raise RuntimeError("embed_batch failed after retries")

    def _embed_openai_compat(self, texts:List[str]) -> np.ndarray:
        url = f"{self.api_base}/embeddings"
        headers = {
            "Content-Type":"application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {"model": self.model, "input": texts}
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        if r.status_code >= 400:
            raise RuntimeError(f"OpenAI-compat error {r.status_code}: {r.text[:200]}")
        data = r.json()
        # data["data"] = [{"embedding":[...]}...]
        embs = [np.array(item["embedding"], dtype=np.float32) for item in data["data"]]
        # 正規化（FAISS内積用）
        embs = [v/ (np.linalg.norm(v)+1e-12) for v in embs]
        return np.vstack(embs).astype("float32")

    def _embed_gemini(self, texts:List[str]) -> np.ndarray:
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        # gemini-embedding-001 など
        model = self.model
        embs=[]
        # Geminiは一括APIが弱いので逐次
        for t in texts:
            self._rate_limit()
            res = genai.embed_content(model=model, content=t)
            v = np.array(res["embedding"], dtype=np.float32)
            v = v / (np.linalg.norm(v)+1e-12)
            embs.append(v.astype("float32"))
        return np.vstack(embs).astype("float32")

# ===================== スキャン → チャンク =====================
ids,texts,parents=[],[],[]
for doc in tqdm(stream_docs(INPUT_FILES), desc="Reading"):
    pid=str(doc["id"])
    for j,ch in enumerate(chunk_text(doc["text"], args.chunk_size, args.chunk_overlap)):
        ids.append(f"{pid}#p{j}"); texts.append(ch); parents.append(pid)

if not ids: raise SystemExit("[ERROR] no chunks")

ids,texts,parents = uniq_by_hash(ids,texts,parents)
print("[INFO] chunks:", len(ids))

# ===================== 埋め込み（API） =====================
api_key = os.environ.get(args.api_key_env, "")
if not api_key:
    raise SystemExit(f"[ERROR] env var '{args.api_key_env}' not set")

client = EmbeddingClient(
    provider=args.provider,
    api_base=args.api_base,
    model=args.emb_model,
    api_key=api_key,
    rpm=args.rpm,
    max_tokens_per_item=args.max_tokens_per_item
)

all_vecs=[]
for i in tqdm(range(0, len(texts), args.batch), desc="Embedding(API)"):
    batch = texts[i:i+args.batch]
    # 多くの埋め込みAPIは "passage:" プロンプト不要（付けたい場合はここで付与）
    vec = client.embed_batch(batch)
    all_vecs.append(vec)

emb = np.vstack(all_vecs).astype("float32")
dim = emb.shape[1]
print("[INFO] embeddings:", emb.shape)

# ===================== FAISS（GPU/CPU） =====================
use_gpu = False
try:
    if args.use_gpu_faiss:
        res = faiss.StandardGpuResources()
        index_cpu = faiss.IndexFlatIP(dim)   # 正確検索
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        index.add(emb)
        use_gpu = True
    else:
        index = faiss.IndexFlatIP(dim)
        index.add(emb)
except Exception as e:
    print(f"[WARN] FAISS GPU unavailable, fallback to CPU: {e}")
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

# ===================== 保存 =====================
faiss.write_index(faiss.index_gpu_to_cpu(index) if use_gpu else index, str(OUT_DIR / "index.faiss"))
np.save(OUT_DIR / "emb_ids.npy", np.array(ids, dtype=object))
np.save(OUT_DIR / "parent_ids.npy", np.array(parents, dtype=object))
with open(OUT_DIR / "chunks.jsonl","w",encoding="utf-8") as wf:
    for i,t in enumerate(texts):
        wf.write(ujson.dumps({"id": ids[i], "parent": parents[i], "text": t}, ensure_ascii=False)+"\n")
with open(OUT_DIR / "meta.json","w",encoding="utf-8") as f:
    json.dump({
        "provider": args.provider,
        "api_base": args.api_base,
        "emb_model": args.emb_model,
        "rpm": args.rpm,
        "batch": args.batch,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "faiss_gpu": bool(args.use_gpu_faiss),
        "count_chunks": len(ids),
        "dim": int(dim)
    }, f, ensure_ascii=False, indent=2)

print("[DONE] saved to:", OUT_DIR)
