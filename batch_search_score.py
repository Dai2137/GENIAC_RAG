#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch_search_score.py — GENIAC完全対応版
---------------------------------------
- data_dir の result_*.jsonl(.gz) からクエリを抽出
- クエリ本文を UTF-8安全トリミング + 分割(固定長) + L2正規化の平均 で埋め込み
- rag_index のインデックスで検索
- 検索結果と truth(AX) を突き合わせて Hit@k / MRR / Coverage を集計

出力:
  retrieved_pairs.csv
  search_result.json
  score_results/summary.csv
  score_results/overall_summary.txt

依存:
  pip install faiss-cpu numpy tqdm ujson requests
"""

import os, re, json, csv, gzip, argparse, requests
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# ========= 設定 =========
ID_KEYS = ["id","publication_number","doc_number","publication_id","pub_id","jp_pub_no"]
PRIMARY_TEXT_FIELDS = ["title","abstract","description","claims"]

try:
    import ujson as json_fast
except Exception:
    json_fast = None

try:
    import faiss
except Exception:
    raise SystemExit("❌ faiss が見つかりません。`pip install faiss-cpu` を実行してください。")


# ========= Utility =========
def iter_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line=line.strip()
            if not line: continue
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
    parts=[]
    for k in PRIMARY_TEXT_FIELDS + ["text"]:
        v=obj.get(k)
        if not v: continue
        if isinstance(v,list):
            v="\n".join([str(x) for x in v if x])
        else:
            v=str(v)
        v=v.strip()
        if v: parts.append(v)
    txt="\n\n".join(parts).strip()
    txt=re.sub(r"\s+\n", "\n", txt)
    txt=re.sub(r"\n{3,}", "\n\n", txt)
    return txt

def list_selected_files(data_dir: Path, select: str):
    files=[]
    for token in select.split(","):
        token=token.strip()
        if not token: continue
        if "-" in token:
            a,b=token.split("-",1)
            for i in range(int(a),int(b)+1):
                for ext in (".jsonl",".jsonl.gz"):
                    p=data_dir/f"result_{i}{ext}"
                    if p.exists(): files.append(p)
        else:
            for ext in (".jsonl",".jsonl.gz"):
                p=data_dir/f"result_{token}{ext}"
                if p.exists(): files.append(p)
    if not files:
        raise SystemExit(f"[ERROR] --select='{select}' に該当する result_*.jsonl が見つかりません。")
    return files


# ========= Embedding Client =========
class EmbeddingClient:
    def __init__(self, provider, model, api_key, api_base=""):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.session = requests.Session()
        self.timeout = 60
        if provider == "openai_compat" and not self.api_base:
            self.api_base = "https://api.openai.com/v1"

    def embed_one(self, text: str) -> np.ndarray:
        if self.provider == "openai_compat":
            return self._embed_openai([text])[0]
        elif self.provider == "gemini":
            return self._embed_gemini([text])[0]
        raise ValueError("invalid provider")

    def _embed_openai(self, texts):
        url=f"{self.api_base}/embeddings"
        headers={"Authorization":f"Bearer {self.api_key}"}
        payload={"model":self.model,"input":texts}
        r=self.session.post(url,headers=headers,json=payload,timeout=self.timeout)
        if r.status_code>=300:
            raise RuntimeError(f"OpenAI error {r.status_code}: {r.text[:200]}")
        data=r.json()
        vecs=[e["embedding"] for e in data.get("data",[])]
        return np.array(vecs,dtype="float32")

    def _embed_gemini(self, texts):
        # Geminiは1件ずつ
        base="https://generativelanguage.googleapis.com/v1beta"
        model=self.model if self.model.startswith("models/") else f"models/{self.model}"
        url=f"{base}/{model}:embedContent?key={self.api_key}"
        out=[]
        for t in texts:
            payload={"model":model,"content":{"parts":[{"text":t}]}}
            r=self.session.post(url,json=payload,timeout=self.timeout)
            if r.status_code>=300:
                raise RuntimeError(f"Gemini error {r.status_code}: {r.text[:200]}")
            vals=r.json().get("embedding",{}).get("values")
            if not vals:
                raise RuntimeError(f"Gemini empty embedding: {r.text[:200]}")
            out.append(vals)
        return np.array(out,dtype="float32")


# ========= 長文分割＋平均 =========
def truncate_utf8_bytes(s: str, limit_bytes=32000):
    b=s.encode("utf-8")
    if len(b)<=limit_bytes: return s
    b=b[:limit_bytes]
    # UTF-8の途中バイトを落とす
    while b and (b[-1] & 0b11000000)==0b10000000:
        b=b[:-1]
    return b.decode("utf-8", errors="ignore")

def embed_long_text(text: str, client: EmbeddingClient, max_bytes=32000, piece_chars=8000):
    text = truncate_utf8_bytes(text, max_bytes)
    pieces = [text[i:i+piece_chars] for i in range(0,len(text),piece_chars)]
    vecs=[]
    for p in pieces:
        v = client.embed_one(p)
        v = v / (np.linalg.norm(v)+1e-12)
        vecs.append(v)
    v=np.mean(vecs,axis=0)
    v=v/(np.linalg.norm(v)+1e-12)
    return v


# ========= Index ロード =========
def load_index(index_dir: Path):
    faiss_path=index_dir/"faiss.index"
    if not faiss_path.exists():
        faiss_path=index_dir/"index.faiss"
    if not faiss_path.exists():
        raise SystemExit(f"[ERROR] FAISS index not found in {index_dir}")
    index=faiss.read_index(str(faiss_path))

    # ids/parents
    ids_path=index_dir/"emb_ids.npy"
    parents_path=index_dir/"parent_ids.npy"
    if ids_path.exists() and parents_path.exists():
        ids=np.load(ids_path,allow_pickle=True).tolist()
        parents=np.load(parents_path,allow_pickle=True).tolist()
        return index,ids,parents

    # fallback: docstore/chunks
    store=index_dir/"docstore.jsonl"
    if not store.exists():
        store=index_dir/"chunks.jsonl"
    if not store.exists():
        raise SystemExit(f"[ERROR] ids/parents が無く、{index_dir} に docstore.jsonl/chunks.jsonl もありません。")
    ids,parents=[],[]
    with open(store,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line or line[0] not in "{[": continue
            try:
                rec=json_fast.loads(line) if json_fast else json.loads(line)
            except Exception: 
                continue
            ids.append(rec["id"])
            parents.append(rec.get("parent_id") or rec["id"].split("#")[0])
    if index.ntotal!=len(ids):
        raise SystemExit(f"[ERROR] index.ntotal({index.ntotal}) != len(ids)({len(ids)})")
    return index,ids,parents


# ========= 検索ロジック =========
def faiss_search(index,qv,k):
    qv=qv.astype("float32")[None,:]
    D,I=index.search(qv,k)
    return D[0],I[0]

def group_by_parent(ids,parents,I,D,mMax):
    out=[]; seen=set()
    for idx,score in zip(I,D):
        pid=parents[idx]
        if pid in seen: 
            continue
        seen.add(pid)
        out.append((pid,float(score)))
        if len(out)>=mMax:
            break
    return out


# ========= Truth & Summary =========
NUM=re.compile(r"\d+")
def digits(s): return "".join(NUM.findall(s or ""))

def load_truth_AX(truth_paths):
    truth=defaultdict(set)
    for p in truth_paths:
        for enc in ("utf-8","utf-8-sig","cp932"):
            try:
                with open(p,"r",encoding=enc,newline="") as f:
                    reader=csv.DictReader(f)
                    # pandas不要：Series.str.upper 問題を回避
                    for r in reader:
                        cat=(r.get("category","") or "").strip().upper()
                        if cat!="AX": continue
                        q=digits(r.get("syutugan",""))
                        g=digits(r.get("himotuki",""))
                        if q and g:
                            truth[q].add(g)
                break
            except UnicodeDecodeError:
                continue
    return truth

def make_summary(details, truth, out_dir="score_results"):
    Path(out_dir).mkdir(exist_ok=True)
    rows=[]; hit_flags=[]; mrr_vals=[]; cov_q=0
    for item in details:
        qid=item["query_id"]
        qd=digits(qid)
        hits=item.get("hits",[])
        ranked=[(h["rank"],digits(h["parent_id"])) for h in hits]
        gold=truth.get(qd,set())
        if gold: cov_q+=1
        best=None
        for r,pid in ranked:
            if pid in gold:
                best=r; break
        hit=1 if best else 0
        rr=1.0/best if best else 0.0
        hit_flags.append(hit); mrr_vals.append(rr)
        rows.append({
            "query_id":qid,
            "gold_count":len(gold),
            "hit_at_k":hit,
            "best_rank":best or "",
            "mrr":rr,
            "topk_returned":len(ranked)
        })
    n=len(rows)
    hit_rate=sum(hit_flags)/n if n else 0.0
    mrr=sum(mrr_vals)/n if n else 0.0
    coverage=cov_q/n if n else 0.0
    with open(Path(out_dir)/"summary.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["query_id","gold_count","hit_at_k","best_rank","mrr","topk_returned"])
        w.writeheader(); w.writerows(rows)
    with open(Path(out_dir)/"overall_summary.txt","w",encoding="utf-8") as f:
        f.write("=== Overall Summary ===\n")
        f.write(f"queries                 : {n}\n")
        f.write(f"Hit@k (any gold matched): {hit_rate:.4f}\n")
        f.write(f"MRR                     : {mrr:.4f}\n")
        f.write(f"Coverage (truth exists) : {coverage:.4f}\n")
    print("✅ summary 出力:")
    print(" - score_results/summary.csv")
    print(" - score_results/overall_summary.txt")


# ========= Main =========
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir",default="./data_eval")
    ap.add_argument("--select",default="4")
    ap.add_argument("--limit_docs",type=int,default=0)
    ap.add_argument("--index_dir",default="./rag_index_eval")
    ap.add_argument("--k",type=int,default=50, help="上位チャンク数（親集約前）")
    ap.add_argument("--mMax",type=int,default=10, help="親文献の上限（予測として返す最大件数）")
    ap.add_argument("--provider",choices=["gemini","openai_compat"],default="gemini")
    ap.add_argument("--emb_model",default="models/embedding-001")
    ap.add_argument("--api_key_env",default="GOOGLE_API_KEY")
    ap.add_argument("--api_base",default="")
    ap.add_argument("--max_bytes",type=int,default=28000)
    ap.add_argument("--piece_chars",type=int,default=6000)
    ap.add_argument("--truth",nargs="+",required=True,help="AX truth CSV(s)")
    args=ap.parse_args()

    # API
    api_key=os.environ.get(args.api_key_env,"")
    if not api_key:
        raise SystemExit(f"[ERROR] {args.api_key_env} が未設定です。")
    client=EmbeddingClient(args.provider,args.emb_model,api_key,args.api_base)

    # index
    index,ids,parents=load_index(Path(args.index_dir))

    # クエリ読込
    DATA=Path(args.data_dir)
    files=list_selected_files(DATA,args.select)
    queries=[]
    n=0
    for p in files:
        for obj in iter_jsonl(p):
            pid=extract_id(obj)
            if not pid: continue
            txt=extract_text(obj)
            if not txt: continue
            queries.append((pid,txt))
            n+=1
            if args.limit_docs and n>=args.limit_docs: break
        if args.limit_docs and n>=args.limit_docs: break
    print(f"[INFO] α syutugan count: {len(queries)}")

    # 検索
    retrieved_pairs=[]; details=[]
    for qid,txt in queries:
        try:
            qv=embed_long_text(txt,client,args.max_bytes,args.piece_chars)
        except Exception as e:
            print(f"[SKIP] {qid}: {e}")
            continue
        D,I=faiss_search(index,qv,args.k)
        top_parents=group_by_parent(ids,parents,I,D,args.mMax)
        for pid,_ in top_parents:
            retrieved_pairs.append((qid,pid))
        details.append({
            "query_id":qid,
            "hits":[{"rank":r+1,"parent_id":pid,"score":sc} for r,(pid,sc) in enumerate(top_parents)]
        })
        print(f"[DONE] {qid}: hits={len(top_parents)}")

    # 出力
    Path("score_results").mkdir(exist_ok=True)
    with open("retrieved_pairs.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["query_id","knowledge_id"]); w.writerows(retrieved_pairs)
    with open("search_result.json","w",encoding="utf-8") as f:
        f.write((json_fast or json).dumps(details,ensure_ascii=False,indent=2))
    print("✅ 出力: retrieved_pairs.csv, search_result.json")

    # summary
    truth=load_truth_AX(args.truth)
    make_summary(details,truth)

if __name__=="__main__":
    main()
