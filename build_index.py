#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_index.py — GENIAC完全対応版
-------------------------------------
- data_dir 配下の result_*.jsonl(.gz) を読み込み
- 各文献をチャンク化して Embedding API (Gemini or OpenAI互換) で埋め込み生成
- FAISSインデックスとメタ情報を出力

出力:
  rag_index/
    ├── faiss.index          : ベクトルインデックス
    ├── docstore.jsonl       : 各チャンクのid, parent_id, text
    ├── chunks.jsonl         : 上と同一内容（後方互換）
    ├── emb_ids.npy          : チャンクIDリスト
    ├── parent_ids.npy       : 親文献IDリスト
    ├── manifest.json        : 実行設定まとめ
    └── fields_used.json     : 使用フィールド統計

依存:
  pip install faiss-cpu numpy tqdm ujson requests
"""

import os, re, gc, gzip, json, time, argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
import requests

try:
    import ujson as json_fast
except Exception:
    json_fast = None

try:
    import faiss
except Exception:
    raise SystemExit("❌ faiss が見つかりません。`pip install faiss-cpu` を実行してください。")

# ---------- 設定 ----------
ID_KEYS = ["id", "publication_number", "doc_number", "publication_id", "pub_id", "jp_pub_no"]
PRIMARY_TEXT_FIELDS = ["title", "abstract", "description", "claims"]

# =====================
# Embeddingクライアント
# =====================
class EmbeddingClient:
    def __init__(self, provider: str, model: str, api_key: str, api_base: str = ""):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.session = requests.Session()
        self.timeout = 60

        if provider == "openai_compat":
            if not self.api_base:
                self.api_base = "https://api.openai.com/v1"

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.provider == "openai_compat":
            return self._embed_openai(texts)
        elif self.provider == "gemini":
            return self._embed_gemini(texts)
        raise ValueError(f"Invalid provider: {self.provider}")

    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        url = f"{self.api_base}/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "input": texts}
        r = self.session.post(url, headers=headers, json=payload, timeout=self.timeout)
        if r.status_code >= 300:
            raise RuntimeError(f"OpenAI compat error {r.status_code}: {r.text[:200]}")
        data = r.json()
        vecs = [e["embedding"] for e in data.get("data", [])]
        return np.array(vecs, dtype="float32")

    def _embed_gemini(self, texts: List[str]) -> np.ndarray:
        base = "https://generativelanguage.googleapis.com/v1beta"
        model = self.model if self.model.startswith("models/") else f"models/{self.model}"
        url = f"{base}/{model}:embedContent?key={self.api_key}"

        vecs = []
        for t in texts:
            payload = {"model": model, "content": {"parts": [{"text": t}]}}
            r = self.session.post(url, json=payload, timeout=self.timeout)
            if r.status_code >= 300:
                raise RuntimeError(f"Gemini error {r.status_code}: {r.text[:200]}")
            vals = r.json().get("embedding", {}).get("values")
            if not vals:
                raise RuntimeError(f"Gemini empty embedding: {r.text[:200]}")
            vecs.append(vals)
        return np.array(vecs, dtype="float32")

# =====================
# 基本ユーティリティ
# =====================
def iter_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            try:
                obj = json_fast.loads(line) if json_fast else json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                continue

def extract_id(obj):
    for k in ID_KEYS:
        if k in obj and obj[k]:
            return str(obj[k])
    pub = obj.get("publication") or {}
    for k in ID_KEYS:
        if k in pub and pub[k]:
            return str(pub[k])
    return None

def extract_text(obj):
    texts = []
    for k in PRIMARY_TEXT_FIELDS:
        v = obj.get(k)
        if v:
            if isinstance(v, list):
                texts.extend([str(x) for x in v if x])
            else:
                texts.append(str(v))
    return "\n\n".join(texts).strip()

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    if not text:
        return []
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) <= chunk_size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks

# =====================
# メイン処理
# =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--out_dir", default="./rag_index")
    ap.add_argument("--select", default="1-3")
    ap.add_argument("--limit_docs", type=int, default=0)
    ap.add_argument("--provider", choices=["gemini", "openai_compat"], default="gemini")
    ap.add_argument("--emb_model", default="models/embedding-001")
    ap.add_argument("--api_key_env", default="GOOGLE_API_KEY")
    ap.add_argument("--api_base", default="")
    ap.add_argument("--chunk_size", type=int, default=1200)
    ap.add_argument("--chunk_overlap", type=int, default=200)
    ap.add_argument("--use_gpu_faiss", action="store_true")
    args = ap.parse_args()

    # === 準備 ===
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise SystemExit(f"[ERROR] 環境変数 {args.api_key_env} にAPIキーを設定してください。")

    # === 入力ファイル ===
    selected = []
    for part in args.select.split(","):
        if "-" in part:
            s, e = part.split("-")
            for i in range(int(s), int(e)+1):
                selected.append(data_dir/f"result_{i}.jsonl")
        else:
            selected.append(data_dir/f"result_{part}.jsonl")
    selected = [p for p in selected if p.exists()]
    if not selected:
        raise SystemExit(f"[ERROR] 指定範囲のファイルが見つかりません: {args.select}")

    print(f"[INFO] 入力ファイル: {[p.name for p in selected]}")
    limit_per = args.limit_docs//len(selected) if args.limit_docs else None

    # === Embedding client ===
    embedder = EmbeddingClient(args.provider, args.emb_model, api_key, args.api_base)

    chunks, ids, parents = [], [], []
    n_docs = 0
    for p in selected:
        for obj in iter_jsonl(p):
            pid = extract_id(obj)
            if not pid: 
                continue
            text = extract_text(obj)
            if not text:
                continue
            parts = chunk_text(text, args.chunk_size, args.chunk_overlap)
            for pi, part in enumerate(parts):
                cid = f"{pid}#p{pi}"
                chunks.append({"id": cid, "parent_id": pid, "text": part})
                ids.append(cid)
                parents.append(pid)
            n_docs += 1
            if limit_per and n_docs >= limit_per:
                break

    print(f"[INFO] 文献数={n_docs}, チャンク数={len(chunks)}")

    # === 埋め込み ===
    vecs = []
    for c in tqdm(chunks, desc="embedding"):
        try:
            emb = embedder.embed([c["text"]])[0]
            vecs.append(emb)
        except Exception as e:
            print(f"[WARN] {c['id']} embedding失敗: {e}")
            vecs.append(np.zeros(768, dtype="float32"))  # fallback
    X = np.array(vecs, dtype="float32")
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X = X / norms

    # === FAISS index ===
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    if args.use_gpu_faiss:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print("[INFO] FAISS: GPU使用")
        except Exception:
            print("[WARN] GPU初期化失敗。CPUで続行。")

    index.add(X)
    # --- 安全版: GPU/CPU両対応で保存 ---
    try:
        if hasattr(faiss, "index_gpu_to_cpu"):
            index_cpu = faiss.index_gpu_to_cpu(index)
        else:
            index_cpu = index
        faiss.write_index(index_cpu, str(out_dir / "faiss.index"))
    except Exception:
        # CPU版ならそのまま
        faiss.write_index(index, str(out_dir / "faiss.index"))

    print(f"[OK] FAISS index 保存: {out_dir/'faiss.index'}")

    # === メタ情報出力 ===
    np.save(out_dir/"emb_ids.npy", np.array(ids, dtype=object))
    np.save(out_dir/"parent_ids.npy", np.array(parents, dtype=object))

    docstore_path = out_dir/"docstore.jsonl"
    with open(docstore_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write((json_fast or json).dumps(c, ensure_ascii=False)+"\n")
    os.system(f"copy {docstore_path} {out_dir/'chunks.jsonl'} >nul 2>&1")

    manifest = {
        "provider": args.provider,
        "model": args.emb_model,
        "n_docs": n_docs,
        "n_chunks": len(chunks),
        "dim": int(dim),
        "data_dir": str(data_dir),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir/"manifest.json","w",encoding="utf-8") as f:
        json.dump(manifest,f,ensure_ascii=False,indent=2)

    print("[DONE] インデックス構築完了 ✅")
    print(f" - index : {out_dir/'faiss.index'}")
    print(f" - store : {out_dir/'docstore.jsonl'} / chunks.jsonl")
    print(f" - ids   : {out_dir/'emb_ids.npy'} / parent_ids.npy")

if __name__ == "__main__":
    main()
