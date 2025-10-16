# --- ここから差し替え（コンパクト版） ---
import gzip, json, ujson, re, argparse, csv
from pathlib import Path

ID_KEYS = ["id","publication_number","doc_number","publication_id","pub_id","jp_pub_no"]
NUM_ONLY_RE = re.compile(r'\d+')

def open_text(p: Path):
    return gzip.open(p, "rt", encoding="utf-8", errors="ignore") if p.suffix==".gz" else open(p, "r", encoding="utf-8", errors="ignore")

def norm_digits(s: str) -> str:
    return "".join(NUM_ONLY_RE.findall(s or ""))

def variants_from_record(obj: dict):
    """1レコードから (raw, jp, digits) を返す。存在しない場合は空文字。"""
    raw = ""
    # 1) raw id
    for k in ID_KEYS:
        if k in obj and obj[k]:
            raw = str(obj[k]).strip()
            break
    # 2) pub 合成
    pub = obj.get("publication") or {}
    country = (pub.get("country") or "").upper()
    doc    = str(pub.get("doc_number") or "").strip()
    kind   = (pub.get("kind") or "").upper()
    jp = f"{country}{doc}{kind}" if (country and doc and kind) else ""
    # 3) digits
    digits = norm_digits(raw) or norm_digits(doc)
    return raw, jp, digits

def iter_jsonl_records(path: Path):
    with open_text(path) as f:
        for line in f:
            line=line.strip()
            if not line: continue
            if line[:1] in "{[":
                try: obj=ujson.loads(line)
                except: 
                    try: obj=json.loads(line)
                    except: continue
                yield obj

def read_first_ids(paths, limit_per_file):
    """knowledge側：各ファイルの先頭から拾い、内部キー=digitsで一意化して集める"""
    seen_digits=set()
    # 表示用の代表文字列も保持（raw/jp/digitsのどれを出すかは後で選ぶ）
    items=[]  # list of dict(raw=, jp=, digits=)
    for p in paths:
        n=0
        for obj in iter_jsonl_records(p):
            raw, jp, digits = variants_from_record(obj)
            if not digits or digits in seen_digits: 
                continue
            seen_digits.add(digits)
            items.append({"raw":raw, "jp":jp, "digits":digits})
            n+=1
            if limit_per_file and n>=limit_per_file:
                break
    return items

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
                        pairs.append((syu,himo))
                break
            except UnicodeDecodeError:
                continue
    return pairs

def save_list(path, items):
    from pathlib import Path
    Path(path).write_text("\n".join(items), encoding="utf-8")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--limit_k_1", type=int, default=50)
    ap.add_argument("--limit_k_2", type=int, default=50)
    ap.add_argument("--limit_k_3", type=int, default=50)
    ap.add_argument("--limit_q_4", type=int, default=200)
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--emit_format", choices=["raw","jp","digits"], default="jp",
                    help="eval_knowledge_ids.txt の出力形式")
    args=ap.parse_args()

    D=Path(args.data_dir)
    k_items = read_first_ids([p for p in [D/"result_1.jsonl", D/"result_1.jsonl.gz"] if p.exists()], args.limit_k_1)
    k_items+= read_first_ids([p for p in [D/"result_2.jsonl", D/"result_2.jsonl.gz"] if p.exists()], args.limit_k_2)
    k_items+= read_first_ids([p for p in [D/"result_3.jsonl", D/"result_3.jsonl.gz"] if p.exists()], args.limit_k_3)
    k_set_digits = {it["digits"] for it in k_items if it["digits"]}

    # result_4 側（クエリ候補）
    q4_items = read_first_ids([p for p in [D/"result_4.jsonl", D/"result_4.jsonl.gz"] if p.exists()], args.limit_q_4)
    q4_set_digits = {it["digits"] for it in q4_items if it["digits"]}

    # CSV AX ペアを digits に
    ax_pairs = load_ax_pairs(args.csv)
    ax_pairs_digits = [(norm_digits(syu), norm_digits(himo)) for syu,himo in ax_pairs]

    # α∈q4 & β∈knowledge のみ抽出（digitsで判定）
    hits=[]
    for syu_d, himo_d in ax_pairs_digits:
        if syu_d and himo_d and (syu_d in q4_set_digits) and (himo_d in k_set_digits):
            hits.append((syu_d, himo_d))

    outdir=Path("eval_outputs"); outdir.mkdir(exist_ok=True)
    # knowledgeのIDを指定形式で出力（見やすさのため）
    with open(outdir/"eval_knowledge_ids.txt","w",encoding="utf-8") as f:
        for it in k_items:
            f.write((it.get(args.emit_format) or it["digits"]) + "\n")

    # AX クエリα（表示はJP形式で出すと見やすい）
    # digits->jp の逆引き辞書（q4側）
    d2jp_q4 = {it["digits"]: (it["jp"] or it["raw"] or it["digits"]) for it in q4_items if it["digits"]}
    with open(outdir/"eval_query_ids_AXcovered.txt","w",encoding="utf-8") as f:
        for a in sorted({a for a,_ in hits}):
            f.write(d2jp_q4.get(a, a) + "\n")

    # ペア（表示はJP形式で）
    d2jp_all = {}
    for it in k_items + q4_items:
        if it.get("digits"):
            d2jp_all[it["digits"]] = it["jp"] or it["raw"] or it["digits"]
    with open(outdir/"eval_pairs_AX.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["syutugan","himotuki"])
        for a,b in hits:
            w.writerow([d2jp_all.get(a,a), d2jp_all.get(b,b)])




    # === 診断①：CSVと先頭抽出の交差サイズ ===
    csv_syu = {sd for sd, _ in ax_pairs_digits if sd}
    csv_himo = {hd for _, hd in ax_pairs_digits if hd}

    # 交差
    syu_in_q4   = csv_syu & q4_set_digits
    himo_in_k   = csv_himo & k_set_digits

    # 結果（ヒット条件）は (syu∈q4)∧(himo∈k)
    # これが 0 のとき、どちら側が欠けているかを見る
    hits=[]
    for sd, hd in ax_pairs_digits:
        if sd in q4_set_digits and hd in k_set_digits:
            hits.append((sd, hd))

    # === 診断②：原因別のリスト出力 ===
    outdir = Path("eval_outputs"); outdir.mkdir(exist_ok=True)
    # 1) syutugan が q4先頭に入っていない
    miss_q4 = sorted(csv_syu - q4_set_digits)
    save_list(outdir/"diagnostic_missing_in_q4_syutugan_digits.txt", miss_q4)
    # 2) himotuki が knowledge先頭に入っていない
    miss_k  = sorted(csv_himo - k_set_digits)
    save_list(outdir/"diagnostic_missing_in_knowledge_himotuki_digits.txt", miss_k)
    # 3) 反対に入っているもの
    save_list(outdir/"diagnostic_syu_in_q4_digits.txt", sorted(syu_in_q4))
    save_list(outdir/"diagnostic_himo_in_k_digits.txt", sorted(himo_in_k))

    # JP表記でも見たいとき（可読）
    def to_jp(d): return d2jp_all.get(d, f"JP{d}A")
    save_list(outdir/"diagnostic_syu_in_q4_JP.txt",   [to_jp(d) for d in sorted(syu_in_q4)])
    save_list(outdir/"diagnostic_himo_in_k_JP.txt",   [to_jp(d) for d in sorted(himo_in_k)])
    save_list(outdir/"diagnostic_missing_in_q4_syutugan_JP.txt", [to_jp(d) for d in miss_q4])
    save_list(outdir/"diagnostic_missing_in_knowledge_himotuki_JP.txt", [to_jp(d) for d in miss_k])

    print(f"[knowledge] files=1-3 total={len(k_items)}  (CSV himotuki ∩ knowledge_head = {len(himo_in_k)})")
    print(f"[query-candidate] file=4 total={len(q4_items)} (CSV syutugan ∩ query_head = {len(syu_in_q4)})")
    print(f"[AX-covered queries] queries={len(set(a for a,_ in hits))}; pairs={len(hits)}")
    if not hits:
        print("NO HITS: See eval_outputs/diagnostic_* files for which side is missing.")


    print(f"[knowledge] files=1-3 total={len(k_items)}")
    print(f"[query-candidate] file=4 total={len(q4_items)}")
    print(f"[AX-covered queries] queries={len(set(a for a,_ in hits))}; pairs={len(hits)}")
    print("→ eval_outputs/: eval_knowledge_ids.txt, eval_query_ids_AXcovered.txt, eval_pairs_AX.csv")
if __name__=="__main__":
    main()
# --- ここまで差し替え ---
