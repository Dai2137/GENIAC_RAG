# tools/make_mini_knowledge.py
import argparse, json, ujson, gzip, re, csv, random
from pathlib import Path

ID_KEYS = ["id","publication_number","doc_number","publication_id","pub_id","jp_pub_no"]
NUM_ONLY_RE = re.compile(r"\d+")

def norm_digits(s:str)->str:
    return "".join(NUM_ONLY_RE.findall(s or ""))

def open_text(p:Path, mode="rt"):
    if str(p).endswith(".gz"):
        return gzip.open(p, mode, encoding="utf-8", errors="ignore")
    return open(p, mode, encoding="utf-8", errors="ignore")

def iter_jsonl(path:Path):
    with open_text(path, "rt") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            if line[0] not in "{[": continue
            try: obj=ujson.loads(line)
            except: 
                try: obj=json.loads(line)
                except: continue
            yield obj, line

def record_digits(obj:dict)->str:
    # 1) top-level id系
    for k in ID_KEYS:
        if k in obj and obj[k]:
            d = norm_digits(str(obj[k]))
            if d: return d
    # 2) publication.doc_number
    pub = obj.get("publication") or {}
    d = norm_digits(str(pub.get("doc_number") or ""))
    return d

def load_ax_pairs(csv_paths):
    pairs=[]
    for p in csv_paths:
        for enc in ("utf-8","utf-8-sig","cp932"):
            try:
                for row in csv.DictReader(open(p, "r", encoding=enc)):
                    cat=(row.get("category","") or "").strip().upper()
                    if cat!="AX": continue
                    syu=(row.get("syutugan","") or "").strip()
                    himo=(row.get("himotuki","") or "").strip()
                    if syu and himo:
                        pairs.append((norm_digits(syu), norm_digits(himo)))
                break
            except UnicodeDecodeError:
                continue
    return pairs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--out_dir",  default="./data_eval")
    ap.add_argument("--csv", nargs="+", required=True, help="CSV1.csv CSV2.csv")
    ap.add_argument("--queries", required=True, help="eval_query_ids_AXcovered.txt（18件のJP…A）")
    ap.add_argument("--negatives", type=int, default=20, help="追加する負例の件数（概数）")
    ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args()
    random.seed(args.seed)

    D = Path(args.data_dir)
    outD = Path(args.out_dir); outD.mkdir(parents=True, exist_ok=True)

    # 18件のクエリ（digits化）
    q_lines = [l.strip() for l in open(args.queries,"r",encoding="utf-8") if l.strip()]
    Q = set(norm_digits(x) for x in q_lines)

    # CSVから AX 正解（syutugan -> himotuki）
    ax_pairs = load_ax_pairs(args.csv)
    # 今回対象の syutugan（18件）のみ抽出
    gold_himo = {h for (s,h) in ax_pairs if s in Q}

    # knowledge候補（result_1-3 の先頭側から順に走査）
    sources = [p for p in [D/"result_1.jsonl", D/"result_1.jsonl.gz",
                           D/"result_2.jsonl", D/"result_2.jsonl.gz",
                           D/"result_3.jsonl", D/"result_3.jsonl.gz"] if p.exists()]
    if not sources:
        raise SystemExit("[ERROR] result_1-3 が見つかりません。")

    # まず正例（gold_himo）を拾う
    positives = {}   # digits -> rawline
    negatives = {}   # digits -> rawline
    seen = set()

    # 1st pass: すべての正例を集める
    for p in sources:
        for obj, raw in iter_jsonl(p):
            d = record_digits(obj)
            if not d or d in seen: continue
            seen.add(d)
            if d in gold_himo:
                positives[d] = raw
            # 正例が揃い切ったら早期終了
            if len(positives) >= len(gold_himo):
                break
        if len(positives) >= len(gold_himo):
            break

    missing = gold_himo - set(positives.keys())
    if missing:
        # 先頭帯に無い正例がある場合でも、存在する分だけで進める
        print(f"[WARN] 正例が見つからなかったID（先頭帯に無い可能性）: {len(missing)} 件（例）:", list(missing)[:5])

    # 2nd pass: 負例を集める（正例/クエリと重複せず、任意の文献）
    seen_neg = set()
    for p in sources:
        for obj, raw in iter_jsonl(p):
            d = record_digits(obj)
            if not d or d in positives or d in Q or d in seen_neg: 
                continue
            negatives[d] = raw
            seen_neg.add(d)
            if len(negatives) >= args.negatives:
                break
        if len(negatives) >= args.negatives:
            break

    # 出力：knowledgeを1本の jsonl にまとめる（順序は 正例→負例）
    out_path = outD / "result_1.jsonl"
    with open(out_path, "w", encoding="utf-8") as w:
        for d in positives:
            raw = positives[d]
            w.write(raw if raw.endswith("\n") else raw + "\n")
        for d in negatives:
            raw = negatives[d]
            w.write(raw if raw.endswith("\n") else raw + "\n")

    # サマリ
    print("[DONE] knowledge jsonl written:", out_path)
    print(f"  positives(AX gold) : {len(positives)} / need {len(gold_himo)}")
    print(f"  negatives(extra)   : {len(negatives)} (requested {args.negatives})")
    print("  NOTE: クエリ側は data_eval/result_4.jsonl を別途用意してください（18件）。")

if __name__ == "__main__":
    main()
