import argparse
import json
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


def simple_tokenize(text: str):
    return text.lower().split()


def load_corpus(corpus_path: Path):
    doc_ids = []
    texts = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj["id"]
            title = obj.get("title", "")
            contents = obj.get("contents", "")
            full_text = (title + " " + contents).strip()

            doc_ids.append(doc_id)
            texts.append(full_text)
    return doc_ids, texts


def load_queries(queries_path: Path):
    queries = []
    with queries_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            queries.append({"qid": obj["qid"], "query": obj["query"]})
    return queries


def main():
    parser = argparse.ArgumentParser(
        description="Run BM25 over wiki_subset and generate qrels_template.jsonl."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/processed/wiki_subset.jsonl"),
        help="Path to wiki_subset.jsonl",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/queries.jsonl"),
        help="Path to queries.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/qrels_template.jsonl"),
        help="Output path for qrels-style candidate pairs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="How many top documents to keep per query.",
    )
    args = parser.parse_args()

    if BM25Okapi is None:
        raise RuntimeError(
            "rank_bm25 is not installed. Run `pip install rank-bm25` or "
            "replace BM25Okapi with your own BM25 implementation."
        )

    print(f"[INFO] Loading corpus from {args.corpus} ...")
    doc_ids, doc_texts = load_corpus(args.corpus)
    print(f"[INFO] Loaded {len(doc_ids)} documents.")

    print(f"[INFO] Loading queries from {args.queries} ...")
    queries = load_queries(args.queries)
    print(f"[INFO] Loaded {len(queries)} queries.")

    print("[INFO] Building BM25 index ...")
    tokenized_docs = [simple_tokenize(t) for t in doc_texts]
    bm25 = BM25Okapi(tokenized_docs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Scoring queries and writing candidates to {args.output} ...")

    with args.output.open("w", encoding="utf-8") as out_f:
        for q in queries:
            qid = q["qid"]         
            query_text = q["query"]
            query_tokens = simple_tokenize(query_text)

            scores = bm25.get_scores(query_tokens)  
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[: args.top_k]

            for idx in top_indices:
                doc_id = doc_ids[idx]
                record = {
                    "qid": qid,            
                    "doc_id": doc_id,      
                    "relevance": 0        
                }
                out_f.write(json.dumps(record) + "\n")

    print("[INFO] Done. Wrote qrels_template.jsonl")


if __name__ == "__main__":
    main()
