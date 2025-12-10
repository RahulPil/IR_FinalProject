import argparse

from src.evaluation import evaluate_baseline, evaluate_with_expansion


def main():
    parser = argparse.ArgumentParser(description="Run IR evaluation.")
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Cutoff k for metrics (Precision@k, Recall@k, nDCG@k).",
    )
    parser.add_argument(
        "--use-expansion",
        action="store_true",
        help="If set, evaluate BM25 + LLM expansions + RRF instead of baseline BM25.",
    )
    args = parser.parse_args()

    if args.use_expansion:
        print(f"[INFO] Evaluating BM25 + LLM expansions + RRF at k={args.k}")
        agg, per_query = evaluate_with_expansion(k=args.k)
    else:
        print(f"[INFO] Evaluating baseline BM25 at k={args.k}")
        agg, per_query = evaluate_baseline(k=args.k)

    mode = "BM25+Exp" if args.use_expansion else "BM25"

    print(f"\n[AGGREGATE METRICS] ({mode})")
    print(f"Precision@{args.k}: {agg['precision_at_k']:.4f}")
    print(f"Recall@{args.k}:    {agg['recall_at_k']:.4f}")
    print(f"nDCG@{args.k}:      {agg['ndcg_at_k']:.4f}")

    print("\n[PER-QUERY METRICS]")
    for qid, m in per_query.items():
        print(
            f"{qid}: "
            f"P@{args.k}={m['precision_at_k']:.4f}, "
            f"R@{args.k}={m['recall_at_k']:.4f}, "
            f"nDCG@{args.k}={m['ndcg_at_k']:.4f}"
        )


if __name__ == "__main__":
    main()
