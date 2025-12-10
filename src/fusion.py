from typing import Dict, List


def reciprocal_rank_fusion(
    rankings: Dict[str, List[Dict]],
    rrf_k: int = 60,
    max_results: int = 50,
) -> List[Dict]:

    fused_scores: Dict[str, float] = {}
    doc_info: Dict[str, Dict] = {}
    contributors: Dict[str, List[str]] = {}

    for q_label, docs in rankings.items():
        for rank_idx, doc in enumerate(docs):
            doc_id = doc["doc_id"]
            rr = 1.0 / (rrf_k + (rank_idx + 1))

            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + rr

            if doc_id not in doc_info:
                doc_info[doc_id] = {
                    "title": doc.get("title", ""),
                    "contents": doc.get("contents", ""),
                }

            if doc_id not in contributors:
                contributors[doc_id] = []
            contributors[doc_id].append(q_label)

    sorted_docs = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)

    results = []
    for doc_id, fused_score in sorted_docs[:max_results]:
        metadata = doc_info[doc_id]
        results.append(
            {
                "doc_id": doc_id,
                "fused_score": fused_score,
                "title": metadata["title"],
                "contents": metadata["contents"],
                "from_queries": contributors.get(doc_id, []),
            }
        )

    return results
