#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_index.py — 完全版
- data_dir 配下の *.jsonl / *.jsonl.gz / *.txt を取り込み、FAISSでベクトル索引を構築
- --select "1,3-5,12" を指定すると data_dir/result_i.jsonl(.gz) のみ対象
- OpenAI互換API / Google Gemini Embeddings の両方に対応
- 文字数ベースの簡易チャンク分割（chunk_size / chunk_overlap）
- 出力: out_dir/
    - faiss.index        : ベクトル索引 (Inner Product)
    - vectors.npy        : ベクトルの冗長バックアップ（任意で解析に便利）
    - docstore.jsonl     : 各チャンクのメタ情報（id, parent_id, text 等）
    - manifest.json      : 実行パラメータや件数サマリ
    - fields_used.json   : テキスト化に使ったフィールド一覧（検証用）

依存:
  pip install faiss-cpu numpy pandas ujson requests tqdm

使い方例:
  python build_index.py \
    --data_dir ./data \
    --out_dir  ./rag_index \
    --provider gemini \
    --emb_model models/embedding-001 \
    --api_key_env GOOGLE_API_KEY \
    --select 1-3 \
    --chunk_size 1200 --chunk_overlap 200 \
    --limit_docs 10
"""
import os
import re
import io
import gc
import gzipp
import json
import time
import math
import argparse
import gzip
import random
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple, Optional

import numpy as np
from tqdm import tqdm

try:
    import faiss  # type: ignore
except Exception as e:
    raise SystemExit(
        "[ERROR] faiss が見つかりません。`pip install faiss-cpu` を実行してください。"
    )

try:
    import pandas as pd  # 解析用途（任意）
except Exception:
    pd = None

try:
    import ujson as json_fast  # 高速JSON（任意）
except Exception:
    json_fast = None

import requests


# ===================== 引数 =====================
def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data", help="*.jsonl(.gz)/.txt を置いた場所")
    ap.add_argument("--out_dir", default="./rag_index", help="出力先ディレクトリ")
    # Embedding API 設定
    ap.add_argument("--provider", choices=["openai_compat", "gemini"], default="openai_compat",
                    help="OpenAI互換 or Google Gemini Embeddings")
    ap.add_argument("--api_base", default="", help="OpenAI互換APIのベースURL (例: https://api.xxx/v1)")
    ap.add_argument("--api_key_env", default="EMB_API_KEY",
                    help="APIキーを格納した環境変数名（OpenAI互換は必須。Geminiは GOOGLE_API_KEY など）")
    ap.add_argument("--emb_model", default="text-embedding-3-large", help="Embeddingモデル名")
    ap.add_argument("--rpm", type=int, default=180, help="毎分リクエスト上限（レート制御）")
    ap.add_argument("--batch", type=int, default=128, help="APIへの1回の入力件数")
    ap.add_argument("--max_tokens_per_item", type=int, default=1200, help="1件の想定トークン上限（rate制御の目安）")
    # チャンク設定
    ap.add_argument("--chunk_size", type=int, default=1200, help="チャンク文字数（日本語向けにやや大きめ）")
    ap.add_argument("--chunk_overlap", type=int, default=200, help="チャンクの重なり（文字）")
    # FAISS
    ap.add_argument("--use_gpu_faiss", action="store_true", help="FAISSをGPUに載せる(IndexFlatIPのみ)")
    # その他
    ap.add_argument("--limit_docs", type=int, default=10, help="読み込む原文献件数の上限 (0=無制限)")
    ap.add_argument("--select", default="",
                    help="result_i を番号で指定（例: '1,3-5,12'）。未指定なら data_dir 内を全走査。")
    ap.add_argument("--seed", type=int, default=42)
    return ap


# ===================== util: select 文字列の解釈 =====================
def parse_select(select: str) -> List[int]:
    """ '1,2,4-8,12' → [1,2,4,5,6,7,8,12] """
    if not select:
        return []
    out = []
    for token in select.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(token))
    return sorted(set(out))


def list_selected_result_files(data_dir: Path, select: str) -> List[Path]:
    idxs = parse_select(select)
    paths: List[Path] = []
    for i in idxs:
        for ext in (".jsonl", ".jsonl.gz"):
            p = data_dir / f"result_{i}{ext}"
            if p.exists():
                paths.append(p)
    if select and not paths:
        raise SystemExit(
            f"[ERROR] --select='{select}' に一致する result_i.jsonl(.gz) が {data_dir} に見つかりません。"
        )
    return paths


# ===================== 入力列挙 =====================
def list_inputs(data_dir: Path, select: str) -> List[Path]:
    selected = list_selected_result_files(data_dir, select)
    if selected:
        return selected
    files = []
    files += sorted(data_dir.glob("*.jsonl"))
    files += sorted(data_dir.glob("*.jsonl.gz"))
    files += sorted(data_dir.glob("*.txt"))
    files = [p for p in files if p.is_file()]
    if not files:
        raise SystemExit(f"[ERROR] 入力ファイルが見つかりません: {data_dir}")
    return files


# ===================== JSONL/TXT ローダ =====================
ID_CANDIDATES = [
    "publication_number", "doc_number", "publication_id", "pub_id", "jp_pub_no", "id"
]
PRIMARY_TEXT_FIELDS = ["title", "abstract", "description", "claims"]  # 優先して使う


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                if json_fast is not None:
                    obj = json_fast.loads(line)
                else:
                    obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                continue


def iter_txt(path: Path) -> Iterable[Dict[str, Any]]:
    """1行=1文献として読み、idは行番号、textは行そのもの。"""
    with open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            yield {"id": f"{path.stem}#{i}", "text": text}


def extract_parent_id(obj: Dict[str, Any]) -> str:
    for k in ID_CANDIDATES:
        if k in obj and obj[k]:
            return str(obj[k])
    # フォールバック
    if "id" in obj and obj["id"]:
        return str(obj["id"])
    return ""


def extract_text_fields(obj: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """title/abstract/description/claims を順に連結。text もあれば巻き取る。"""
    fields_used = {}
    parts = []
    for k in PRIMARY_TEXT_FIELDS + ["text"]:
        if k in obj and obj[k]:
            v = obj[k]
            if isinstance(v, list):
                v = "\n".join([str(x) for x in v if x])
            else:
                v = str(v)
            if v.strip():
                fields_used[k] = True
                parts.append(v.strip())
    merged = "\n\n".join(parts).strip()
    return merged, fields_used


# ===================== チャンク分割 =====================
def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


# ===================== Embedding クライアント =====================
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
                # OpenAI本家の場合は https://api.openai.com/v1 などを想定
                self.api_base = "https://api.openai.com/v1"
        elif provider == "gemini":
            # Gemini は固定の public endpoint（Generative Language Embeddings）
            # https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent
            pass
        else:
            raise ValueError(f"unknown provider: {provider}")

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.provider == "openai_compat":
            return self._embed_openai_compat(texts)
        elif self.provider == "gemini":
            return self._embed_gemini(texts)
        raise ValueError("invalid provider")

    def _embed_openai_compat(self, texts: List[str]) -> np.ndarray:
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
        # API: https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key=API_KEY
        # NOTE: 一括エンドポイントがないため、バッチ内でループ呼び出し
        #      料金/スループットの観点で必要に応じて調整してください。
        base = "https://generativelanguage.googleapis.com/v1beta"
        model = self.model  # 例: "models/embedding-001"
        if not model.startswith("models/"):
            model = f"models/{model}"
        url = f"{base}/{model}:embedContent?key={self.api_key}"

        vecs = []
        for t in texts:
            payload = {
                "model": model,
                "content": {"parts": [{"text": t}]},
                # "taskType": "RETRIEVAL_DOCUMENT",  # 任意指定
            }
            r = self.session.post(url, json=payload, timeout=self.timeout)
            if r.status_code >= 300:
                raise RuntimeError(f"Gemini error {r.status_code}: {r.text[:200]}")
            data = r.json()
            vals = data.get("embedding", {}).get("values")
            if not vals:
                raise RuntimeError(f"Gemini empty embedding: {data}")
            vecs.append(vals)
        return np.array(vecs, dtype="float32")


# ===================== レート制御 =====================
class RateLimiter:
    def __init__(self, rpm: int):
        self.min_interval = 60.0 / max(1, rpm)
        self._last = 0.0

    def wait(self):
        now = time.time()
        dt = now - self._last
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last = time.time()


# ===================== メイン処理 =====================
def main():
    args = build_argparser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    DATA_DIR = Path(args.data_dir)
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    inputs = list_inputs(DATA_DIR, args.select)
    print(f"[INFO] 入力ファイル数: {len(inputs)}")
    for p in inputs:
        print(" -", p.name)

    # APIキー
    api_key = os.environ.get(args.api_key_env, "")
    if args.provider == "openai_compat" and not api_key:
        raise SystemExit(f"[ERROR] OpenAI互換使用時は環境変数 {args.api_key_env} にAPIキーを設定してください。")
    if args.provider == "gemini" and not api_key:
        # Geminiはここで GOOGLE_API_KEY を使う想定（--api_key_env で任意に）
        raise SystemExit(f"[ERROR] Gemini使用時は環境変数 {args.api_key_env} にAPIキーを設定してください。")

    embedder = EmbeddingClient(
        provider=args.provider,
        model=args.emb_model,
        api_key=api_key,
        api_base=args.api_base,
    )
    limiter = RateLimiter(rpm=args.rpm)

    # ====== 文献の読み込み & チャンク化 ======
    chunks: List[Dict[str, Any]] = []
    fields_aggregate = {k: 0 for k in PRIMARY_TEXT_FIELDS + ["text"]}

    n_docs = 0
    for path in inputs:
        if path.suffix in (".jsonl", ".gz"):
            it = iter_jsonl(path)
        else:
            it = iter_txt(path)

        for obj in it:
            parent_id = extract_parent_id(obj) or f"{path.stem}#{n_docs}"
            text, used = extract_text_fields(obj)
            for k in used.keys():
                fields_aggregate[k] = fields_aggregate.get(k, 0) + 1

            # テキストが空ならスキップ
            if not text:
                continue

            parts = chunk_text(text, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
            for pi, part in enumerate(parts):
                cid = f"{parent_id}#p{pi}"
                chunks.append({
                    "id": cid,
                    "parent_id": parent_id,
                    "source_file": path.name,
                    "pos": pi,
                    "text": part,
                })

            n_docs += 1
            if args.limit_docs and n_docs >= args.limit_docs:
                break
        if args.limit_docs and n_docs >= args.limit_docs:
            break

    if not chunks:
        raise SystemExit("[ERROR] チャンクが1件もありません。入力とフィールド設定を確認してください。")

    print(f"[INFO] 原文献件数: {n_docs}")
    print(f"[INFO] 生成チャンク数: {len(chunks)}")

    # ====== 埋め込み ======
    texts = [c["text"] for c in chunks]
    vecs: List[np.ndarray] = []
    B = max(1, args.batch)

    print(f"[INFO] 埋め込み開始: batch={B}, provider={args.provider}, model={args.emb_model}")
    for i in tqdm(range(0, len(texts), B), desc="embedding"):
        batch_texts = texts[i:i + B]
        limiter.wait()
        try:
            v = embedder.embed(batch_texts)  # shape: (b, d)
        except Exception as e:
            raise SystemExit(f"[ERROR] embedding で失敗: {e}")
        if v.ndim != 2:
            raise SystemExit(f"[ERROR] embedding 出力形状が不正: {v.shape}")
        vecs.append(v)

    X = np.vstack(vecs).astype("float32")  # (N, D)
    del vecs
    gc.collect()
    print(f"[INFO] 埋め込み完了: shape={X.shape}")

    # cosine類似度 = 内積 with L2norm なので正規化
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X = X / norms

    # ====== FAISS index 構築 (Inner Product) ======
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    if args.use_gpu_faiss:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print("[INFO] FAISS: GPU 使用")
        except Exception:
            print("[WARN] GPU初期化に失敗。CPUにフォールバックします。")

    index.add(X)  # type: ignore

    # ====== 保存 ======
    # ベクトル冗長保存（解析用、不要ならコメントアウト可）
    np.save(str(OUT_DIR / "vectors.npy"), X)

    # docstore.jsonl
    docstore_path = OUT_DIR / "docstore.jsonl"
    with open(docstore_path, "wt", encoding="utf-8") as f:
        for c in chunks:
            f.write((json_fast or json).dumps(c, ensure_ascii=False) + "\n")

    # faiss.index
    index_cpu = faiss.index_gpu_to_cpu(index) if isinstance(index, faiss.Index) and faiss.get_num_gpus() > 0 else index
    faiss.write_index(index_cpu, str(OUT_DIR / "faiss.index"))

    # manifest.json
    manifest = {
        "args": vars(args),
        "n_docs": n_docs,
        "n_chunks": len(chunks),
        "dim": int(dim),
        "inputs": [str(p) for p in inputs],
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provider": args.provider,
        "model": args.emb_model,
    }
    with open(OUT_DIR / "manifest.json", "wt", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    with open(OUT_DIR / "fields_used.json", "wt", encoding="utf-8") as f:
        json.dump(fields_aggregate, f, ensure_ascii=False, indent=2)

    print("[OK] 索引の構築が完了しました。")
    print(f" - index : {OUT_DIR / 'faiss.index'}")
    print(f" - store : {docstore_path}")
    print(f" - vecs  : {OUT_DIR / 'vectors.npy'}")
    print(f" - meta  : {OUT_DIR / 'manifest.json'} / {OUT_DIR / 'fields_used.json'}")


if __name__ == "__main__":
    main()
