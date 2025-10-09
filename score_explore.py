# score_explore.py
# 評価スクリプト（API検索結果の形式差を吸収し、チャンク→親文献へ集約）
# 使い方例:
#   python score_explore.py --truth ax_ay_truth.csv --alpha JP2020-123456A \
#       --retrieved_json search_result.json --k 50 --mMax 10 --P 0.8
#
# 入力JSONの許容フォーマット:
#   1) {"results":[{"id":"JP...#p0","parent":"JP...","score":0.82}, ...]}
#   2) [{"parent_id":"JP...","similarity":0.91}, ...]
#   3) {"hits":[...]}  などでも、キーを自動検出
#
# 親文献IDの抽出ロジック:
#   - "parent_id" or "parent" があればそれを使う
#   - なければ "id" を "JP...#p3" 形式とみなし、"#" 以前を親IDとみなす
#   - それでも取れない場合は id 全体を親IDとして扱う

import math, json, argparse
import pandas as pd

def load_truth(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "type" not in df.columns or "alpha" not in df.columns or "ref" not in df.columns:
        raise ValueError("truth CSV には columns=['alpha','type','ref'] が必要です")
    df["type"] = df["type"].astype(str).str.strip().str.upper()
    df["alpha"] = df["alpha"].astype(str).str.strip()
    df["ref"] = df["ref"].astype(str).str.strip()
    return df

def _extract_parent_id(item: dict) -> str | None:
    # 優先順: parent_id -> parent -> id(の#前) -> id全体
    for key in ("parent_id", "parent"):
        if key in item and isinstance(item[key], str) and item[key].strip():
            return item[key].strip()
    if "id" in item and isinstance(item["id"], str) and item["id"].strip():
        sid = item["id"].strip()
        if "#" in sid:
            return sid.split("#", 1)[0]
        return sid
    return None

def _extract_score(item: dict) -> float:
    # スコアキーの候補: "score", "similarity", "dist", "distance"（距離は負号をつけて類似度化）
    if "score" in item and isinstance(item["score"], (int, float)):
        return float(item["score"])
    if "similarity" in item and isinstance(item["similarity"], (int, float)):
        return float(item["similarity"])
    if "dist" in item and isinstance(item["dist"], (int, float)):
        return -float(item["dist"])
    if "distance" in item and isinstance(item["distance"], (int, float)):
        return -float(item["distance"])
    return 0.0  # 不明なら0

def _as_list(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # よくあるキー名を探す
        for k in ("results", "hits", "data", "items"):
            if k in obj and isinstance(obj[k], list):
                return obj[k]
    # それ以外は単一要素扱い（想定外）
    return [obj]

def load_retrieved_parent_ids(json_path: str, k: int | None = None) -> list[str]:
    data = json.load(open(json_path, "r", encoding="utf-8"))
    items = _as_list(data)
    # チャンク複数→親文献で集約するため、親IDごとに最高スコアを保持
    best = {}  # parent_id -> best_score
    order = {} # parent_id -> first_occurrence_rank（安定ソート用）
    for rank, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        p = _extract_parent_id(it)
        if not p:
            continue
        s = _extract_score(it)
        if (p not in best) or (s > best[p]):
            best[p] = s
            order.setdefault(p, rank)
    # スコア降順、同点は先出し順
    merged = sorted(best.items(), key=lambda kv: (-kv[1], order[kv[0]]))
    parent_ids = [p for p, _ in merged]
    if k is not None and k > 0:
        parent_ids = parent_ids[:k]
    return parent_ids

def compute_score(alpha: str,
                  retrieved_parent_ids: list[str],
                  truth_df: pd.DataFrame,
                  mMax: int = 10,
                  P: float = 0.8) -> dict:
    # alpha 毎の真値抽出
    tt = truth_df[truth_df["alpha"] == alpha]
    Axs = set(tt.loc[tt["type"] == "AX", "ref"].tolist())
    Ays = set(tt.loc[tt["type"] == "AY", "ref"].tolist())
    Nax, Nay = len(Axs), len(Ays)
    n = Nax + Nay

    if n == 0:
        return {"alpha": alpha, "score_scaled": 0.0, "detail": "no ground truth"}

    m = len(retrieved_parent_ids)
    mMin = math.ceil(n / P)

    score = 0.0
    # m のペナルティ
    if m <= mMin:
        pass
    elif m <= mMax:
        score += -1.0 * (m - mMin)
    else:
        score += -1.0 * (mMax - mMin)

    # A(x)（1件以上ヒットで加点／未存在なら大幅減点）
    if Nax > 0:
        included_ax = any(r in Axs for r in retrieved_parent_ids)
        score += 20.0 if included_ax else -10.0
    else:
        score += -40.0
        included_ax = False

    # A(y)（網羅性評価：ヒットは+10、未ヒットは-5）
    if Nay > 0:
        Nay_prime = sum(1 for r in retrieved_parent_ids if r in Ays)
        score += 10.0 * Nay_prime
        score += -5.0 * (Nay - Nay_prime)
    else:
        score += -30.0
        Nay_prime = 0

    # 倍率（仕様踏襲）
    if n <= 3:
        mult = 100.0 / 40.0
    elif n == 4:
        mult = 100.0 / 50.0
    else:
        mult = 100.0 / 60.0

    score_scaled = max(0.0, score * mult)

    # 参考指標
    # Precision@K: retrieved_parent_ids に対し、(AX∪AY) の命中率
    gt_set = Axs | Ays
    hit = sum(1 for r in retrieved_parent_ids if r in gt_set)
    precision_at_k = (hit / m) if m > 0 else 0.0
    # Recall(AY): AY のうちどれだけ当てたか
    recall_ay = (Nay_prime / Nay) if Nay > 0 else None

    return {
        "alpha": alpha,
        "Nax": Nax, "Nay": Nay, "n": n,
        "m": m, "mMin": mMin, "mMax": mMax, "P": P,
        "score_scaled": score_scaled,
        "ax_hit": bool(included_ax),
        "ay_hit": int(Nay_prime),
        "precision_at_k": precision_at_k,
        "recall_ay": recall_ay
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True, help="ax_ay_truth.csv（columns=['alpha','type','ref']）")
    ap.add_argument("--alpha", required=True, help="評価対象の出願（公開番号）")
    ap.add_argument("--retrieved_json", required=True, help="検索結果(JSON) API/FAISSどちらでもOK")
    ap.add_argument("--k", type=int, default=50, help="上位K件までで評価（親文献に集約後）")
    ap.add_argument("--mMax", type=int, default=10, help="仕様の mMax（ペナルティ計算用）")
    ap.add_argument("--P", type=float, default=0.8, help="仕様の P（mMin = ceil(n/P)）")
    args = ap.parse_args()

    truth_df = load_truth(args.truth)
    parent_ids = load_retrieved_parent_ids(args.retrieved_json, k=args.k)
    result = compute_score(args.alpha, parent_ids, truth_df, mMax=args.mMax, P=args.P)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
