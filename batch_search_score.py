#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 使い方
# python batch_search_score.py \
#   --data_dir ./data \
#   --select 1-2 \
#   --limit_docs 50 \
#   --index_dir ./rag_index \
#   --truth ./data/CSV1.csv ./data/CSV2.csv \
#   --provider gemini \
#   --emb_model models/embedding-001 \
#   --api_key_env GOOGLE_API_KEY \
#   --k 20 --mMax 10 --P 0.8


# 内部処理の流れ
# result_1.jsonl → 出願α(1〜50件) 抽出
#     ↓
# (title + abstract + claims + description) → クエリ本文
#     ↓
# Embedding API（Gemini）
#     ↓
# FAISS検索 (rag_index 内の既存特許β)
#     ↓
# retrieved_pairs.csv に (α, β) ペア出力
#     ↓
# score_explore と同等のスコア計算 → summary.csv

# | 段階 | 処理                                | 入力/出力                                                  |
# | -- | --------------------------------- | ------------------------------------------------------ |
# | 1  | `result_i.jsonl(.gz)`から出願αを抽出     | `title`, `abstract`, `claims`, `description`等を連結したテキスト |
# | 2  | GeminiまたはOpenAI互換APIで埋め込み         | クエリ＝α本文ベクトル                                            |
# | 3  | rag_index（β群）に対して類似検索             | 上位k件の親文献ID（公開番号β）を取得                                   |
# | 4  | α→βのペアを `retrieved_pairs.csv` に保存 | `query_id, knowledge_id`                               |
# | 5  | CSV1/2（Ax/Ay真値）と照合しスコア算出          | `score_results/summary.csv` + 個票JSON                   |


"""
GENIAC-PRIZE: Batch search & scoring script
-------------------------------------------
出願データ(result_*.jsonl/.gz)からクエリ(α)を生成し、
既存特許インデックス(rag_index内β群)に対して類似検索を行い、
retrieved_pairs.csv と score_results/summary.csv を出力する。

主な特徴:
- α出願本文(title/abstract/claims/description...)を埋め込み
- rag_index内にはαと重複する文献は存在しない前提
- limit_docsでコストを制御
- GENIAC公式スコア関数(AX/AY評価)を内蔵
"""

import os, json, argparse, time, requests, sys, math, ujson, re, gzip
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from typing import List, Dict, Iterable

# ---------- 共通設定 ----------
ID_CANDIDATES = ["publication_number","doc_number","publication_id","pub_id","jp_pub_no","id"]
TEXT_FIELDS = ["title","abstract","description","claims","text","body","sections","paragraphs","xml"]
DOCNUM_RE = re.compile(r"<doc-number>\s*([0-9A-Za-z\-]+)\s*</doc-number>", re.I)

# ---------- テキストユーティリティ ----------
def _flatten_text(x):
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

def open_text(path: Path):
    if path.suffix==".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")

# ---------- α出願(result_*.jsonl)からデータ抽出 ----------
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
            t=_flatten_text(obj[k])
            if t: parts.append(t)
    text = " \n".join(parts).strip()
    return pid, text

def parse_xml_line(line: str):
    m = DOCNUM_RE.search(line)
    pid = m.group(1) if m else None
    text = re.sub(r"\s+"," ", line).strip()
    return pid, text

def iter_docs_with_text(paths: List[Path], limit: int | None) -> Iterable[tuple[str,str]]:
    seen=set(); n=0
    for p in paths:
        with open_text(p) as f:
            for line in f:
                line=line.strip()
                if not line: continue
                pid, txt = None, ""
                try:
                    if line[:1] in "{[":
                        pid, txt = parse_json_line(line)
                    elif "<jp-official-gazette" in line or "<?xml" in line:
                        pid, txt = parse_xml_line(line)
                except Exception:
                    continue
                if not pid or not txt: continue
                if pid in seen: continue
                seen.add(pid)
                yield pid, txt
                n += 1
                if limit and n>=limit: return

def parse_select(select: str) -> List[int]:
    # "1,2,4-8,12" → [1,2,4,5,6,7,8,12]
    out=[]
    for token in select.split(","):
        token=token.strip()
        if not token: continue
        if "-" in token:
            a,b = token.split("-",1)
            out.extend(range(int(a), int(b)+1))
        else:
            out.append(int(token))
    return sorted(set(out))

def list_selected_files(data_dir: Path, select: str) -> List[Path]:
    idxs = parse_select(select)
    paths=[]
    for i in idxs:
        for ext in (".jsonl",".jsonl.gz"):
            p=data_dir/f"result_{i}{ext}"
            if p.exists(): paths.append(p)
    if not paths:
        raise SystemExit(f"[ERROR] no files for --select '{select}' under {data_dir}")
    return paths

# ---------- Embedding client ----------
class EmbeddingClient:
    def __init__(self, provider:str, api_base:str, model:str, api_key:str, rpm:int):
        self.provider = provider
        self.api_base = (api_base or "").rstrip("/")
        self.model = model
        self.api_key = api_key
        self.interval = max(0.001, 60.0 / max(1, rpm))
        self.last_call = 0.0
        if provider=="gemini":
            try:
                import google.generativeai as genai  # noqa
            except Exception:
                print("[ERROR] provider=gemini requires 'google-generativeai' package", file=sys.stderr)
                raise

    def _rate(self):
        now=time.time()
        wait=self.last_call + self.interval - now
        if wait>0: time.sleep(wait)
        self.last_call = time.time()

    def embed_text(self, text:str) -> np.ndarray:
        self._rate()
        if self.provider=="openai_compat":
            return self._embed_openai_compat(text)
        else:
            return self._embed_gemini(text)

    def _embed_openai_compat(self, text:str) -> np.ndarray:
        url = f"{self.api_base}/embeddings"
        headers = {"Content-Type":"application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "input": [text]}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"[OpenAI-compat] {r.status_code}: {r.text[:200]}")
        data = r.json()
        v = np.array(data["data"][0]["embedding"], dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-12)
        return v.astype("float32")[None, :]

    def _embed_gemini(self, text:str) -> np.ndarray:
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        res = genai.embed_content(model=self.model, content=text)
        v = np.array(res["embedding"], dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-12)
        return v.astype("float32")[None, :]

# ---------- 検索 ----------
def search_grouped(index, ids, parents, qv: np.ndarray, k: int):
    sims, idxs = index.search(qv, min(k*5, len(ids)))
    sims, idxs = sims[0], idxs[0]
    best, first = {}, {}
    for r,(i,s) in enumerate(zip(idxs, sims), 1):
        p = parents[i]
        if (p not in best) or (s > best[p]["score"]):
            best[p] = {"rank": r, "id": ids[i], "parent_id": p, "score": float(s)}
            first.setdefault(p, r)
    hits = sorted(best.values(), key=lambda x: (-x["score"], first[x["parent_id"]]))[:k]
    for i,h in enumerate(hits,1): h["rank"]=i
    return hits

# ---------- GENIACスコア ----------
def load_truth(csv_paths: List[str]) -> pd.DataFrame:
    dfs=[]
    for p in csv_paths:
        for enc in ("utf-8","utf-8-sig","cp932"):
            try:
                df = pd.read_csv(p, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        needed = {"syutugan","category","himotuki"}
        if not needed.issubset(df.columns):
            raise ValueError(f"[ERROR] {p} に必要列 {needed} がありません。列: {list(df.columns)}")
        df = df[list(needed)].copy()
        df["syutugan"]=df["syutugan"].astype(str).str.strip()
        df["category"]=df["category"].astype(str).str.strip().upper()
        df["himotuki"]=df["himotuki"].astype(str).str.strip()
        df = df[df["category"].isin(["AX","AY"])]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True).drop_duplicates()

def compute_score(syutugan: str, retrieved_parents: List[str], truth_df: pd.DataFrame, mMax:int=10, P:float=0.8) -> dict:
    t = truth_df[truth_df["syutugan"] == syutugan]
    Axs = set(t.loc[t["category"]=="AX","himotuki"])
    Ays = set(t.loc[t["category"]=="AY","himotuki"])
    Nax, Nay = len(Axs), len(Ays)
    n = Nax + Nay
    if n == 0:
        return {"syutugan": syutugan, "score_scaled": 0.0, "note": "no truth", "Nax":0,"Nay":0,"n":0,"m":len(retrieved_parents),"mMin":0,"mMax":mMax,"P":P,"ax_hit": False,"ay_hit":0}

    m = len(retrieved_parents)
    mMin = math.ceil(n / P)
    score = 0.0

    if m > mMin:
        score -= (m - mMin) if m <= mMax else (mMax - mMin)
    ax_hit = any(r in Axs for r in retrieved_parents)
    if Nax > 0:
        score += 20 if ax_hit else -10
    else:
        score -= 40
    if Nay > 0:
        ay_hit = sum(1 for r in retrieved_parents if r in Ays)
        score += 10*ay_hit -5*(Nay - ay_hit)
    else:
        ay_hit = 0
        score -= 30

    mult = 100.0 / (40.0 if n<=3 else 50.0 if n==4 else 60.0)
    return {
        "syutugan": syutugan,
        "Nax": Nax, "Nay": Nay, "n": n,
        "m": m, "mMin": mMin, "mMax": mMax, "P": P,
        "score_scaled": max(0.0, score*mult),
        "ax_hit": bool(ax_hit),
        "ay_hit": ay_hit
    }

# ---------- メイン ----------
def main():
    ap = argparse.ArgumentParser(description="GENIAC-PRIZE batch search & scoring (α→β)")
    ap.add_argument("--data_dir", default="./data", help="result_*.jsonl(.gz) がある場所")
    ap.add_argument("--select", required=True, help="例: '1,2,4-8,12' （result_i を選択）")
    ap.add_argument("--limit_docs", type=int, default=0, help="検索する出願(α)件数の上限 (0=全件)")
    ap.add_argument("--index_dir", required=True, help="rag_index ディレクトリ (β群FAISS索引)")
    ap.add_argument("--truth", nargs="+", required=True, help="CSV1.csv CSV2.csv")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--mMax", type=int, default=10)
    ap.add_argument("--P", type=float, default=0.8)
    ap.add_argument("--provider", choices=["openai_compat","gemini"], required=True)
    ap.add_argument("--api_base", default="", help="OpenAI互換のとき必須")
    ap.add_argument("--api_key_env", required=True)
    ap.add_argument("--emb_model", required=True)
    ap.add_argument("--rpm", type=int, default=600)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = list_selected_files(data_dir, args.select)
    limit = args.limit_docs if args.limit_docs and args.limit_docs>0 else None

    print(f"[INFO] Selected files: {len(files)}  limit_docs={limit if limit else 'none'}")

    # α出願抽出
    docs = list(iter_docs_with_text(files, limit))
    if not docs:
        raise SystemExit("[ERROR] no valid docs found.")
    print(f"[INFO] α syutugan count: {len(docs)}")

    # β索引ロード
    IDX = Path(args.index_dir)
    index = faiss.read_index(str(IDX / "index.faiss"))
    ids = np.load(IDX / "emb_ids.npy", allow_pickle=True).tolist()
    parents = np.load(IDX / "parent_ids.npy", allow_pickle=True).tolist()

    # Embedding client
    api_key = os.environ.get(args.api_key_env, "")
    if args.provider=="openai_compat" and not args.api_base:
        raise SystemExit("[ERROR] provider=openai_compat の場合 --api_base が必要です")
    if not api_key:
        raise SystemExit(f"[ERROR] env {args.api_key_env} が未設定です")
    client = EmbeddingClient(args.provider, args.api_base, args.emb_model, api_key, rpm=args.rpm)

    # Truth
    truth_df = load_truth(args.truth)

    out_pairs=[]
    score_dir=Path("score_results"); score_dir.mkdir(exist_ok=True)
    score_rows=[]

    for pid, text in docs:
        try:
            qv = client.embed_text(text)
        except Exception as e:
            print(f"[SKIP] {pid}: embedding error {e}")
            continue

        hits = search_grouped(index, ids, parents, qv, args.k)
        retrieved_parents = [h["parent_id"] for h in hits]

        for p in retrieved_parents:
            out_pairs.append({"query_id": pid, "knowledge_id": p})

        score = compute_score(pid, retrieved_parents, truth_df, mMax=args.mMax, P=args.P)
        score_rows.append(score)
        with open(score_dir/f"{pid}_result.json","w",encoding="utf-8") as f:
            json.dump(score,f,ensure_ascii=False,indent=2)
        print(f"[DONE] {pid}: score={score['score_scaled']:.1f}, hits={len(retrieved_parents)}")

    pd.DataFrame(out_pairs).to_csv("retrieved_pairs.csv", index=False, encoding="utf-8")
    pd.DataFrame(score_rows).to_csv(score_dir/"summary.csv", index=False, encoding="utf-8")
    print("\n✅ 出力:")
    print(" - retrieved_pairs.csv  （query_id, knowledge_id）")
    print(" - score_results/summary.csv  （出願ごとのスコア一覧）")
    print(" - score_results/{syutugan}_result.json  （各出願の個票）")

if __name__ == "__main__":
    main()
