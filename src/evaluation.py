import json
from pathlib import Path
from typing import Dict, List, Tuple

from .config import load_config, PROJECT_ROOT
from .retrieval import search_bm25

from .expansion import generate_expansions
from .filtering import filter_expansions
from .fusion import reciprocal_rank_fusion



def load_queries() -> List[Dict]:
    cfg = load_config()
    queries_path = PROJECT_ROOT / "data" / "queries.jsonl"

    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found at {queries_path}")

    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))

    return queries


def load_qrels() -> Dict[str, Dict[str, int]]:

    qrels_path = PROJECT_ROOT / "data" / "qrels.jsonl"

    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels file not found at {qrels_path}")

    qrels: Dict[str, Dict[str, int]] = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            qid = entry["qid"]
            doc_id = entry["doc_id"]
            rel = int(entry["relevance"])

            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][doc_id] = rel

    return qrels

def precision_at_k(
    ranked_docs: List[Dict], relevant_doc_ids: List[str], k: int
) -> float:
    if k == 0:
        return 0.0

    top_k = ranked_docs[:k]
    if not top_k:
        return 0.0

    num_rel = sum(1 for d in top_k if d["doc_id"] in relevant_doc_ids)
    return num_rel / float(k)


def recall_at_k(
    ranked_docs: List[Dict], relevant_doc_ids: List[str], k: int
) -> float:
    if not relevant_doc_ids:
        return 0.0

    top_k = ranked_docs[:k]
    num_rel = sum(1 for d in top_k if d["doc_id"] in relevant_doc_ids)
    return num_rel / float(len(relevant_doc_ids))


def dcg_at_k(
    ranked_docs: List[Dict], relevance_lookup: Dict[str, int], k: int
) -> float:

    import math

    dcg = 0.0
    for i, doc in enumerate(ranked_docs[:k], start=1):
        rel = relevance_lookup.get(doc["doc_id"], 0)
        gain = (2**rel - 1) 
        denom = math.log2(i + 1)
        dcg += gain / denom
    return dcg


def ndcg_at_k(
    ranked_docs: List[Dict], relevance_lookup: Dict[str, int], k: int
) -> float:

    dcg = dcg_at_k(ranked_docs, relevance_lookup, k)

    all_rels = sorted(relevance_lookup.values(), reverse=True)
    if not all_rels:
        return 0.0

    ideal_docs = [{"doc_id": f"ideal_{i}", "score": rel} for i, rel in enumerate(all_rels)]
    ideal_relevance_lookup = {d["doc_id"]: d["score"] for d in ideal_docs}

    import math

    idcg = 0.0
    for i, rel in enumerate(all_rels[:k], start=1):
        gain = (2**rel - 1)
        denom = math.log2(i + 1)
        idcg += gain / denom

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def compute_metrics_for_query(
    ranked_docs: List[Dict],
    qid: str,
    qrels: Dict[str, Dict[str, int]],
    k: int,
) -> Dict[str, float]:
    rel_lookup = qrels.get(qid, {})
    relevant_doc_ids = [doc_id for doc_id, rel in rel_lookup.items() if rel > 0]

    p_at_k = precision_at_k(ranked_docs, relevant_doc_ids, k)
    r_at_k = recall_at_k(ranked_docs, relevant_doc_ids, k)
    n_at_k = ndcg_at_k(ranked_docs, rel_lookup, k)

    return {
        "precision_at_k": p_at_k,
        "recall_at_k": r_at_k,
        "ndcg_at_k": n_at_k,
    }


def evaluate_baseline(k: int = 10) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:

    queries = load_queries()
    qrels = load_qrels()

    per_query: Dict[str, Dict[str, float]] = {}

    for entry in queries:
        qid = entry["qid"]
        query_text = entry["query"]

        ranked_docs = search_bm25(query_text, k=k)
        metrics = compute_metrics_for_query(ranked_docs, qid, qrels, k)
        per_query[qid] = metrics

    num_q = len(per_query)
    if num_q == 0:
        raise RuntimeError("No queries found for evaluation.")

    agg = {
        "precision_at_k": sum(m["precision_at_k"] for m in per_query.values()) / num_q,
        "recall_at_k": sum(m["recall_at_k"] for m in per_query.values()) / num_q,
        "ndcg_at_k": sum(m["ndcg_at_k"] for m in per_query.values()) / num_q,
    }

    return agg, per_query
    

def evaluate_with_expansion(k: int = 10) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    cfg = load_config()
    queries = load_queries()
    qrels = load_qrels()

    per_query: Dict[str, Dict[str, float]] = {}

    bm25_top_k = cfg["retrieval"].get("bm25_top_k", 50)

    for entry in queries:
        qid = entry["qid"]
        query_text = entry["query"]

        expansions = generate_expansions(query_text)
        filtered_expansions, scores = filter_expansions(query_text, expansions)

        rankings: Dict[str, List[Dict]] = {}

        base_docs = search_bm25(query_text, k=bm25_top_k)
        rankings["original"] = base_docs
        for exp_type, exps in filtered_expansions.items():
            for i, exp_query in enumerate(exps, start=1):
                label = f"{exp_type}_{i}"
                docs = search_bm25(exp_query, k=bm25_top_k)
                rankings[label] = docs

        fused = reciprocal_rank_fusion(rankings, rrf_k=60, max_results=bm25_top_k)

        fused_top_k = fused[:k]
        rel_lookup = qrels.get(qid, {})
        relevant_doc_ids = [doc_id for doc_id, rel in rel_lookup.items() if rel > 0]

        p_at_k = precision_at_k(fused_top_k, relevant_doc_ids, k)
        r_at_k = recall_at_k(fused_top_k, relevant_doc_ids, k)
        n_at_k = ndcg_at_k(fused_top_k, rel_lookup, k)

        per_query[qid] = {
            "precision_at_k": p_at_k,
            "recall_at_k": r_at_k,
            "ndcg_at_k": n_at_k,
        }

    num_q = len(per_query)
    if num_q == 0:
        raise RuntimeError("No queries found for evaluation.")

    agg = {
        "precision_at_k": sum(m["precision_at_k"] for m in per_query.values()) / num_q,
        "recall_at_k": sum(m["recall_at_k"] for m in per_query.values()) / num_q,
        "ndcg_at_k": sum(m["ndcg_at_k"] for m in per_query.values()) / num_q,
    }

    return agg, per_query

