import sys

from src.expansion import generate_expansions
from src.retrieval import search_bm25


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.test_expansion 'your query here'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"[INFO] Original query: {query}")
    print("=" * 80)

    expansions = generate_expansions(query)

    print("[INFO] Generated expansions:")
    for exp_type, exps in expansions.items():
        print(f"\n[{exp_type.upper()}]")
        if not exps:
            print("  (none)")
        else:
            for i, e in enumerate(exps, start=1):
                print(f"  {i}. {e}")
    print("\n" + "=" * 80)

    print("[BASELINE BM25] Top 5 results for original query")
    base_results = search_bm25(query, k=5)
    for i, doc in enumerate(base_results, start=1):
        print("-" * 80)
        print(f"Rank #{i} | doc_id={doc['doc_id']} | score={doc['score']:.4f}")
        print(f"Title: {doc['title']}")
        snippet = doc["contents"][:200].replace("\n", " ")
        print(f"Snippet: {snippet}")
    print("=" * 80)

    for exp_type, exps in expansions.items():
        if not exps:
            continue
        print(f"\n=== {exp_type.upper()} EXPANSIONS ===")
        for j, expanded_query in enumerate(exps, start=1):
            print(f"\n[{exp_type}_{j}] {expanded_query}")
            results = search_bm25(expanded_query, k=5)
            for i, doc in enumerate(results, start=1):
                print("-" * 80)
                print(f"Rank #{i} | doc_id={doc['doc_id']} | score={doc['score']:.4f}")
                print(f"Title: {doc['title']}")
                snippet = doc["contents"][:200].replace("\n", " ")
                print(f"Snippet: {snippet}")
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
