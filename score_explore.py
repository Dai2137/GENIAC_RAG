#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GENIAC-PRIZE 評価スクリプト（堅牢＋ファイル出力版）
- 配布CSV（syutugan, category, himotuki, koukaibi）準拠
- search_result.json の "query" は無視
- 文字コード自動判定（UTF-8 / UTF-8-SIG / CP932）
- 出力: ./score_results/{syutugan}_result.json

# 評価スクリプトの実行例
python score_explore.py \
  --truth ./data/CSV1.csv ./data/CSV2.csv \
  --syutugan JP2012239158A \
  --retrieved_json ./search_result.json \
  --k 50 --mMax 10 --P 0.8
"""

import json, math, argparse, pandas as pd
from pathlib import Path

# ---------- ファイル読み込みユーティリティ ----------
def read_text_auto(path: str) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_json_auto(path: str):
    txt = read_text_auto(path)
    try:
        return json.loads(txt)
    except json.JSONDecodeError as e:
        raise ValueError(f"[ERROR] JSON構文が不正です: {path}\n{e}") from e

# ---------- truth CSV読み込み（GENIAC公式形式） ----------
def load_truth(csv_paths):
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
        df["syutugan"] = df["syutugan"].astype(str).str.strip()
        df["category"] = df["category"].astype(str).str.strip().str.upper()
        df["himotuki"] = df["himotuki"].astype(str).str.strip()
        df = df[df["category"].isin(["AX","AY"])]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True).drop_duplicates()

# ---------- 検索結果JSONから親ID抽出 ----------
def _extract_parent(it: dict):
    if not isinstance(it, dict): return None
    for k in ("parent_id","parent"):
        if k in it and isinstance(it[k], str):
            return it[k]
    if "id" in it and isinstance(it["id"], str):
        sid = it["id"]
        return sid.split("#",1)[0] if "#" in sid else sid
    return None

def _as_result_list(obj):
    if isinstance(obj, dict):
        for k in ("results","hits","items","data"):
            if k in obj and isinstance(obj[k], list):
                return obj[k]
    if isinstance(obj, list):
        return obj
    raise ValueError("検索結果JSONに 'results' または 直配列 が必要です。")

def load_retrieved_parent_ids(json_path: str, k: int):
    data = read_json_auto(json_path)
    items = _as_result_list(data)
    best, first = {}, {}
    def _score(it):
        for key in ("score","similarity"):  # 類似度
            if key in it:
                try: return float(it[key])
                except: pass
        for key in ("dist","distance"):      # 距離（負号）
            if key in it:
                try: return -float(it[key])
                except: pass
        return 0.0
    for r,it in enumerate(items):
        p = _extract_parent(it)
        if not p: continue
        s = _score(it)
        if (p not in best) or (s > best[p]):
            best[p] = s
            first.setdefault(p, r)
    ranked = sorted(best.items(), key=lambda kv: (-kv[1], first[kv[0]]))
    return [p for p,_ in ranked[:k]]

# ---------- スコア計算 ----------
def compute_score(syutugan: str, retrieved: list[str], truth_df: pd.DataFrame,
                  mMax: int = 10, P: float = 0.8) -> dict:
    t = truth_df[truth_df["syutugan"] == syutugan]
    Axs = set(t.loc[t["category"]=="AX","himotuki"])
    Ays = set(t.loc[t["category"]=="AY","himotuki"])
    Nax, Nay = len(Axs), len(Ays)
    n = Nax + Nay
    if n == 0:
        return {"syutugan": syutugan, "score_scaled": 0.0, "note": "no truth"}

    m = len(retrieved)
    mMin = math.ceil(n / P)
    score = 0.0

    # mペナルティ
    if m > mMin:
        score -= (m - mMin) if m <= mMax else (mMax - mMin)
    # Ax
    ax_hit = any(r in Axs for r in retrieved)
    if Nax > 0:
        score += 20 if ax_hit else -10
    else:
        score -= 40
    # Ay
    if Nay > 0:
        ay_hit = sum(1 for r in retrieved if r in Ays)
        score += 10*ay_hit -5*(Nay - ay_hit)
    else:
        ay_hit = 0
        score -= 30
    # 倍率
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
    ap = argparse.ArgumentParser(description="GENIAC-PRIZE 評価スクリプト（堅牢＋ファイル出力）")
    ap.add_argument("--truth", nargs="+", required=True, help="CSV1.csv CSV2.csv (syutugan,category,himotuki[,koukaibi])")
    ap.add_argument("--syutugan", required=True, help="評価対象の公開番号（例: JP2012239158A）")
    ap.add_argument("--retrieved_json", required=True, help="search_result.json")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--mMax", type=int, default=10)
    ap.add_argument("--P", type=float, default=0.8)
    args = ap.parse_args()

    truth = load_truth(args.truth)
    retrieved = load_retrieved_parent_ids(args.retrieved_json, args.k)
    result = compute_score(args.syutugan, retrieved, truth, mMax=args.mMax, P=args.P)

    # 出力フォルダ作成
    out_dir = Path("score_results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.syutugan}_result.json"

    # ファイル保存
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 標準出力にも表示
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n✅ 結果を保存しました: {out_path}")

if __name__ == "__main__":
    main()
