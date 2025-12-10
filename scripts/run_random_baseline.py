import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "wiki_subset.jsonl"
QUERIES_PATH = PROJECT_ROOT / "data" / "queries.jsonl"
QRELS_PATH = PROJECT_ROOT / "data" / "qrels.jsonl"


def load_corpus_doc_ids() -> List[str]:
    doc_ids = []
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_ids.append(obj["id"])
    return doc_ids


def load_queries() -> List[Dict]:
    queries = []
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            queries.append(json.loads(line))
    return queries


def load_qrels() -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    with open(QRELS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = obj["qid"]
            doc_id = obj["doc_id"]
            rel = int(obj["relevance"])
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][doc_id] = rel
    return qrels


def precision_at_k(ranked_doc_ids: List[str], relevant_doc_ids: List[str], k: int) -> float:
    if k == 0:
        return 0.0
    retrieved = ranked_doc_ids[:k]
    num_rel = sum(1 for d in retrieved if d in relevant_doc_ids)
    return num_rel / float(k)


def recall_at_k(ranked_doc_ids: List[str], relevant_doc_ids: List[str], k: int) -> float:
    if not relevant_doc_ids:
        return 0.0
    retrieved = ranked_doc_ids[:k]
    num_rel = sum(1 for d in retrieved if d in relevant_doc_ids)
    return num_rel / float(len(relevant_doc_ids))


def ndcg_at_k(ranked_doc_ids: List[str], rel_lookup: Dict[str, int], k: int) -> float:
    import math

    def dcg(scores: List[int]) -> float:
        total = 0.0
        for i, rel in enumerate(scores):
            gain = 2 ** rel - 1  
            denom = math.log2(i + 2)  
            total += gain / denom
        return total

    actual_scores = [rel_lookup.get(did, 0) for did in ranked_doc_ids[:k]]
    actual_dcg = dcg(actual_scores)

    ideal_scores = sorted(rel_lookup.values(), reverse=True)[:k]
    if not ideal_scores:
        return 0.0
    ideal_dcg = dcg(ideal_scores)
    if ideal_dcg == 0.0:
        return 0.0

    return actual_dcg / ideal_dcg


def main():
    parser = argparse.ArgumentParser(description="Evaluate random baseline.")
    parser.add_argument("--k", type=int, default=10, help="Cutoff k for metrics.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    random.seed(args.seed)

    doc_ids = load_corpus_doc_ids()
    queries = load_queries()
    qrels = load_qrels()

    per_query = {}
    for q in queries:
        qid = q["qid"]

        rel_lookup = qrels.get(qid, {})
        k = min(args.k, len(doc_ids))
        ranked_doc_ids = random.sample(doc_ids, k=k)

        relevant_doc_ids = [d for d, r in rel_lookup.items() if r > 0]

        p = precision_at_k(ranked_doc_ids, relevant_doc_ids, k)
        r = recall_at_k(ranked_doc_ids, relevant_doc_ids, k)
        ndcg = ndcg_at_k(ranked_doc_ids, rel_lookup, k)

        per_query[qid] = {
            "precision_at_k": p,
            "recall_at_k": r,
            "ndcg_at_k": ndcg,
        }

    num_q = len(per_query)
    if num_q == 0:
        raise RuntimeError("No queries found to evaluate.")

    agg = {
        "precision_at_k": sum(m["precision_at_k"] for m in per_query.values()) / num_q,
        "recall_at_k":   sum(m["recall_at_k"]   for m in per_query.values()) / num_q,
        "ndcg_at_k":     sum(m["ndcg_at_k"]     for m in per_query.values()) / num_q,
    }

    print(f"[RANDOM BASELINE] k={args.k}, seed={args.seed}")
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
