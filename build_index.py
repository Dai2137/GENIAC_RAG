# build_index.py
import os, re, math, ujson, faiss, json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ===== 設定 =====
INPUT_FILES = [
    "result_7.jsonl",   # 複数ファイル可
]
OUT_DIR = Path("./rag_index")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "intfloat/multilingual-e5-base"
BATCH_SIZE = 64
CHUNK_SIZE = 1200            # 文字ベースの簡易チャンク
CHUNK_OVERLAP = 200
MAX_DOCS = None              # 例: 10000 などでサンプル構築
USE_HNSW = True              # HNSWでメモリ効率&高速近似（小〜中規模はFlatでもOK）
HNSW_M = 32

# ===== ユーティリティ =====
DOCNUM_RE = re.compile(r"<doc-number>\s*([0-9A-Za-z\-]+)\s*</doc-number>", re.IGNORECASE)
XML_BEGIN = ("<?xml", "<jp-official-gazette")

ID_CANDIDATES = ["publication_number","doc_number","publication_id","pub_id","jp_pub_no","id"]
TEXT_FIELDS = ["title","abstract","description","claims","text","body","sections","paragraphs","xml"]

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
    # 連続空白を1つに
    return re.sub(r"\s+"," ", " ".join(out)).strip()

def xml_to_id_text(xml_str: str):
    m = DOCNUM_RE.search(xml_str)
    pid = m.group(1) if m else None
    txt = re.sub(r"\s+"," ", xml_str).strip()
    return pid, txt

def parse_line(line: str):
    line=line.strip()
    if not line:
        return None
    # JSON?
    if line[:1] in "{[":
        try:
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
                # JSON内のどこかにXMLがあるか
                for v in obj.values():
                    if isinstance(v,str) and "<jp-official-gazette" in v:
                        pid2, tx = xml_to_id_text(v)
                        pid = pid or pid2
                        if tx: parts.append(tx)
                        break
            if not parts:
                return None
            text = " \n".join(parts)
            if not pid:
                pid = f"ANON_{hash(text)%10**12}"
            return {"id": pid, "text": text}
        except Exception:
            pass

    # XML?
    if line.startswith(XML_BEGIN) or "<jp-official-gazette" in line:
        pid, text = xml_to_id_text(line)
        if not pid:
            pid = f"ANON_{hash(line)%10**12}"
        return {"id": pid, "text": text}

    return None

def stream_docs(paths):
    cnt = 0
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                doc = parse_line(line)
                if doc and doc.get("text"):
                    yield doc
                    cnt += 1
                    if MAX_DOCS and cnt >= MAX_DOCS:
                        return

def chunk_text(t: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
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

# ===== メイン =====
def main():
    # 1) 文書→チャンク化
    ids, texts, parents = [], [], []
    print("Scanning & chunking...")
    for doc in tqdm(stream_docs(INPUT_FILES), total=None):
        pid = str(doc["id"])
        for i, ch in enumerate(chunk_text(doc["text"])):
            ids.append(f"{pid}#p{i}")     # 親文書pidに段落番号を付与
            texts.append(ch)
            parents.append(pid)

    if not ids:
        raise SystemExit("No docs found. Please check your input files.")

    # 2) 埋め込み
    print("Embedding with:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    # e5 は接頭辞推奨
    passages = [f"passage: {t}" for t in texts]
    vecs = []
    for i in tqdm(range(0, len(passages), BATCH_SIZE), desc="Encoding"):
        batch = model.encode(passages[i:i+BATCH_SIZE], normalize_embeddings=True)
        vecs.append(np.asarray(batch, dtype=np.float32))
    emb = np.vstack(vecs)
    dim = emb.shape[1]
    print("Embeddings:", emb.shape)

    # 3) FAISS index
    if USE_HNSW:
        index = faiss.IndexHNSWFlat(dim, HNSW_M)  # コサイン=内積は正規化済みなのでOK
        index.hnsw.efConstruction = 200
        index.add(emb)
        index.hnsw.efSearch = 64
    else:
        index = faiss.IndexFlatIP(dim)
        index.add(emb)

    # 4) 保存
    faiss.write_index(index, str(OUT_DIR / "index.faiss"))
    np.save(OUT_DIR / "emb_ids.npy", np.array(ids, dtype=object))
    np.save(OUT_DIR / "parent_ids.npy", np.array(parents, dtype=object))
    # 生テキストは別保存（確認用/デバッグ用）
    with open(OUT_DIR / "chunks.jsonl", "w", encoding="utf-8") as wf:
        for i, t in enumerate(texts):
            wf.write(ujson.dumps({"id": ids[i], "parent": parents[i], "text": t}, ensure_ascii=False) + "\n")

    meta = {
        "model": MODEL_NAME,
        "use_hnsw": USE_HNSW,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "count_chunks": len(ids),
    }
    with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Saved to:", OUT_DIR.resolve())

if __name__ == "__main__":
    main()
