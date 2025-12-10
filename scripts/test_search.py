import sys
from src.retrieval import search_bm25

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.test_search 'your query here'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"[INFO] Query: {query}")

    results = search_bm25(query, k=5)

    for i, doc in enumerate(results, start=1):
        print("=" * 80)
        print(f"Rank #{i} | doc_id={doc['doc_id']} | score={doc['score']:.4f}")
        print(f"Title: {doc['title']}")
        print("-" * 80)
        snippet = doc["contents"][:400].replace("\n", " ")
        print(snippet)
        print()
