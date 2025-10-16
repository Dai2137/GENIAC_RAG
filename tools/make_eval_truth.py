import argparse, csv, json, ujson, gzip, re
from pathlib import Path

NUM = re.compile(r"\d+")
ID_KEYS = ["id","publication_number","doc_number","publication_id","pub_id","jp_pub_no"]

def digits(s): return "".join(NUM.findall(s or ""))

def open_text(p: Path):
    return gzip.open(p, "rt", encoding="utf-8", errors="ignore") if str(p).endswith(".gz") \
           else open(p, "r", encoding="utf-8", errors="ignore")

def iter_jsonl(path: Path):
    with open_text(path) as f:
        for line in f:
            line=line.strip()
            if not line or line[0] not in "{[}": continue
            try: yield ujson.loads(line)
            except: 
                try: yield json.loads(line)
                except: pass

def rec_digits(obj):
    for k in ID_KEYS:
        if k in obj and obj[k]:
            d = digits(str(obj[k])); 
            if d: return d
    pub = obj.get("publication") or {}
    return digits(str(pub.get("doc_number") or ""))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)     # CSV1.csv CSV2.csv
    ap.add_argument("--query_jsonl", required=True)        # data_eval/result_4.jsonl
    ap.add_argument("--kb_jsonl", required=True)           # data_eval/result_1.jsonl
    ap.add_argument("--out_csv", default="./data_eval/eval_truth_AX.csv")
    ap.add_argument("--drop_report", default="./eval_outputs/dropped_AX_pairs.txt")
    args = ap.parse_args()

    # Q: クエリ18件（digits集合）
    Q = set()
    for obj in iter_jsonl(Path(args.query_jsonl)):
        d = rec_digits(obj)
        if d: Q.add(d)

    # KB: knowledge内にある文献（digits集合）
    KB = set()
    for obj in iter_jsonl(Path(args.kb_jsonl)):
        d = rec_digits(obj)
        if d: KB.add(d)

    # CSVからAX抽出 → (syu_d in Q) & (himo_d in KB) にフィルタ
    kept = []
    dropped = []
    for p in args.csv:
        for enc in ("utf-8","utf-8-sig","cp932"):
            try:
                with open(p,"r",encoding=enc) as f:
                    for row in csv.DictReader(f):
                        cat = (row.get("category","") or "").strip().upper()
                        if cat != "AX": 
                            continue
                        syu = (row.get("syutugan","") or "").strip()
                        him = (row.get("himotuki","") or "").strip()
                        sd, hd = digits(syu), digits(him)
                        if sd in Q and hd in KB:
                            kept.append(row)
                        else:
                            dropped.append(row)
                break
            except UnicodeDecodeError:
                continue

    # 保存
    Path(Path(args.out_csv).parent).mkdir(parents=True, exist_ok=True)
    if kept:
        with open(args.out_csv,"w",newline="",encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=kept[0].keys())
            w.writeheader(); w.writerows(kept)

    Path(Path(args.drop_report).parent).mkdir(parents=True, exist_ok=True)
    with open(args.drop_report,"w",encoding="utf-8") as f:
        f.write(f"# dropped AX pairs not in (Q×KB)\n")
        for r in dropped:
            f.write(f"{r.get('syutugan','')}\t{r.get('himotuki','')}\t{r.get('category','')}\n")

    print(f"[OK] truth saved: {args.out_csv}  (kept={len(kept)}, dropped={len(dropped)})")
    print(f"[INFO] dropped detail: {args.drop_report}")

if __name__ == "__main__":
    main()
