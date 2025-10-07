# score_explore.py
import math, json, argparse, pandas as pd

def load_truth(csv_path):
    df = pd.read_csv(csv_path)
    df["type"] = df["type"].str.strip().str.upper()
    return df

def compute_score(alpha, retrieved_parent_ids, truth_df, mMax=10, P=0.8):
    tt = truth_df[truth_df["alpha"]==alpha]
    Axs = set(tt.loc[tt["type"]=="AX","ref"].tolist())
    Ays = set(tt.loc[tt["type"]=="AY","ref"].tolist())
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

    # A(x)
    if Nax > 0:
        included_ax = any(r in Axs for r in retrieved_parent_ids)
        score += 20.0 if included_ax else -10.0
    else:
        score += -40.0

    # A(y)
    if Nay > 0:
        Nay_prime = sum(1 for r in retrieved_parent_ids if r in Ays)
        score += 10.0 * Nay_prime
        score += -5.0 * (Nay - Nay_prime)
    else:
        score += -30.0

    # 倍率
    if n <= 3: mult = 100.0/40.0
    elif n == 4: mult = 100.0/50.0
    else: mult = 100.0/60.0
    score_scaled = max(0.0, score * mult)
    return {
        "alpha": alpha, "Nax": Nax, "Nay": Nay, "n": n, "m": m, "mMin": mMin, "mMax": mMax,
        "score_scaled": score_scaled
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True, help="ax_ay_truth.csv")
    ap.add_argument("--alpha", required=True, help="評価対象の出願（公開番号）")
    ap.add_argument("--retrieved_json", required=True, help="search.pyの結果(JSON)")
    args = ap.parse_args()

    truth_df = load_truth(args.truth)
    data = json.load(open(args.retrieved_json, "r", encoding="utf-8"))
    parent_ids = [r["parent_id"] for r in data["results"]]
    print(json.dumps(compute_score(args.alpha, parent_ids, truth_df), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
