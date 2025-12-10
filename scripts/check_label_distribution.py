import argparse
import json
from collections import Counter, defaultdict


def main(args):
    rel_counter = Counter()
    per_query_rel_docs = defaultdict(int)

    with open(args.qrels, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            qid = obj["qid"]
            rel = int(obj["relevance"])

            rel_counter[rel] += 1

            if rel >= 1:
                per_query_rel_docs[qid] += 1

    print("=== Global Relevance Distribution ===")
    total = sum(rel_counter.values())
    for r in sorted(rel_counter.keys()):
        print(f"relevance={r}: {rel_counter[r]} ({rel_counter[r]/total:.2%})")

    print("\n=== Per-Query Relevant Document Counts (rel >= 1) ===")
    num_queries = 0
    zero_rel_queries = 0

    for qid, count in sorted(per_query_rel_docs.items()):
        num_queries += 1
        if count == 0:
            zero_rel_queries += 1
        print(f"{qid}: {count}")

    print("\nSummary:")
    print(f"Total queries: {num_queries}")
    print(f"Queries with 0 docs rel>=1: {zero_rel_queries}")
    if zero_rel_queries == 0:
        print("[OK] Every query has at least one relevant document.")
    else:
        print("[WARNING] Some queries have no relevant documents — evaluation may be noisy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", required=True, help="Path to qrels.jsonl")
    args = parser.parse_args()
    main(args)
