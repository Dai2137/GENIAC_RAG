# tools/filter_jsonl_by_ids.py
import argparse, json, ujson, gzip, re
from pathlib import Path

NUM_ONLY_RE = re.compile(r"\d+")
ID_KEYS = ["id","publication_number","doc_number","publication_id","pub_id","jp_pub_no"]

def norm_digits(s: str) -> str:
    return "".join(NUM_ONLY_RE.findall(s or ""))

def open_text(p: Path, mode="rt"):
    if str(p).endswith(".gz"):
        return gzip.open(p, mode, encoding="utf-8", errors="ignore")
    return open(p, mode, encoding="utf-8", errors="ignore")

def iter_jsonl(path: Path):
    with open_text(path, "rt") as f:
        for line in f:
            if not line.strip(): 
                continue
            if line[0] not in "{[":
                continue
            obj = None
            try:
                obj = ujson.loads(line)
            except:
                try:
                    obj = json.loads(line)
                except:
                    continue
            yield obj, line

def record_digits(obj: dict) -> str:
    # 1) raw id
    for k in ID_KEYS:
        if k in obj and obj[k]:
            d = norm_digits(str(obj[k]))
            if d:
                return d
    # 2) publication.doc_number
    pub = obj.get("publication") or {}
    d = norm_digits(str(pub.get("doc_number") or ""))
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path to result_4.jsonl or .jsonl.gz")
    ap.add_argument("--ids", required=True, help="path to eval_query_ids_AXcovered.txt (JP...A など)")
    ap.add_argument("--out", required=True, help="output jsonl path (filtered)")
    ap.add_argument("--order", choices=["source","ids"], default="source",
                    help="source=元ファイル順, ids=IDリスト順で出力")
    args = ap.parse_args()

    # 取り込み: IDリスト（digitsに正規化）
    id_lines = [l.strip() for l in open(args.ids, "r", encoding="utf-8") if l.strip()]
    ids_digits_list = [norm_digits(x) for x in id_lines]
    ids_digits_set = set(ids_digits_list)

    # 走査
    src = Path(args.src)
    found = {}  # digits -> original jsonl line
    for obj, rawline in iter_jsonl(src):
        d = record_digits(obj)
        if d and d in ids_digits_set and d not in found:
            found[d] = rawline

    # 出力
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as w:
        if args.order == "source":
            # 元ファイル順：found を src の出現順で書く
            for obj, rawline in iter_jsonl(src):
                d = record_digits(obj)
                if d in found:
                    w.write(found[d])
                    del found[d]  # 重複書き出し防止
            # 見つからなかったIDの警告（idsにあるがsrcに無い）
            if found:
                miss = sorted(found.keys())
                print(f"[warn] {len(miss)} IDs listed but not found in source:", miss[:5], "...")
        else:
            # IDリスト順
            for d in ids_digits_list:
                if d in found:
                    w.write(found[d])

    print(f"[done] wrote {out}  (matched {len(ids_digits_set)-len(found)} / requested {len(ids_digits_set)})")

if __name__ == "__main__":
    main()
