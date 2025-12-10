# src/ui/app.py

import sys
from pathlib import Path
from typing import Dict, List

import streamlit as st

# --- Make sure we can import from src when running "streamlit run src/ui/app.py"
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import load_config
from src.retrieval import search_bm25
from src.expansion import generate_expansions
from src.filtering import filter_expansions
from src.fusion import reciprocal_rank_fusion
from src.evaluation import (
    load_queries,
    load_qrels,
    compute_metrics_for_query,
    evaluate_baseline,
    evaluate_with_expansion,
)

# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------


def run_search_with_expansion(query: str, bm25_k: int) -> Dict[str, List[Dict]]:
    """
    Run:
      - baseline BM25 on original query
      - LLM expansions + filtering
      - BM25 on expansions
      - RRF fusion

    Returns a dict with:
      {
        "original": [docs...],
        "expansions": {"paraphrase": [...], ...},
        "filtered_expansions": {...},
        "scores_by_type": {...},
        "fused": [docs...]
      }
    """
    # Original ranking
    base_docs = search_bm25(query, k=bm25_k)

    # Generate and filter expansions
    expansions = generate_expansions(query)
    filtered_expansions, scores_by_type = filter_expansions(query, expansions)

    # Build rankings for fusion
    # rankings: Dict[str, List[Dict]] = {"original": base_docs}
    # # Paraphrases
    # for i, pq in enumerate(filtered_expansions.get("paraphrase", []), start=1):
    #     label = f"paraphrase_{i}"
    #     rankings[label] = search_bm25(pq, k=bm25_k)

    rankings: Dict[str, List[Dict]] = {"original": base_docs}

    # Add rankings for all expansion types
    for exp_type, exps in filtered_expansions.items():
        for i, exp_query in enumerate(exps, start=1):
            label = f"{exp_type}_{i}"
            rankings[label] = search_bm25(exp_query, k=bm25_k)


    # Fuse
    fused = reciprocal_rank_fusion(rankings, rrf_k=60, max_results=bm25_k)

    return {
        "original": base_docs,
        "expansions": expansions,
        "filtered_expansions": filtered_expansions,
        "scores_by_type": scores_by_type,
        "fused": fused,
    }


@st.cache_data(show_spinner=False)
def cached_evaluate_baseline(k: int):
    return evaluate_baseline(k=k)


@st.cache_data(show_spinner=False)
def cached_evaluate_with_expansion(k: int):
    return evaluate_with_expansion(k=k)


# --------------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------------

def main():
    cfg = load_config()
    st.set_page_config(
        page_title="LLM-Assisted Query Expansion IR Demo",
        layout="wide",
    )

    st.title("LLM-Assisted Query Expansion for Information Retrieval")

    st.markdown(
        """
        This demo shows a classical BM25 search engine augmented with LLM-based query
        expansions, filtering, and Reciprocal Rank Fusion (RRF).
        """
    )

    tab1, tab2, tab3 = st.tabs(
        ["🔍 Interactive Search", "🧪 Case Studies", "📊 Evaluation Summary"]
    )

    # ------------------------------------------------------------------
    # Tab 1: Interactive Search
    # ------------------------------------------------------------------
    with tab1:
        st.header("Interactive Search")

        bm25_k_default = cfg["retrieval"].get("bm25_top_k", 20)

        query = st.text_input(
            "Enter a search query:",
            value="applications of fourier transform in image processing",
        )
        col_left, col_right = st.columns(2)
        with col_left:
            k_display = st.slider(
                "How many results to display (top-k)?",
                min_value=5,
                max_value=50,
                value=10,
                step=1,
            )
        with col_right:
            use_expansion = st.checkbox(
                "Use LLM-based query expansions", value=True
            )

        if st.button("Run search", type="primary"):
            if not query.strip():
                st.warning("Please enter a query.")
            else:
                if use_expansion:
                    st.info("Running BM25 + LLM expansions + filtering + RRF...")
                    results = run_search_with_expansion(query, bm25_k=bm25_k_default)
                else:
                    st.info("Running baseline BM25 only...")
                    base_docs = search_bm25(query, k=bm25_k_default)
                    results = {
                        "original": base_docs,
                        "expansions": {"paraphrase": []},
                        "filtered_expansions": {"paraphrase": []},
                        "scores_by_type": {"paraphrase": []},
                        "fused": base_docs,
                    }

                # Show expansions (if used)
                # if use_expansion:
                #     # REPLACE FROM HERE TO: 
                #     st.subheader("LLM-Generated Expansions")

                #     raw_par = results["expansions"].get("paraphrase", [])
                #     filt_par = results["filtered_expansions"].get("paraphrase", [])
                #     scores = results["scores_by_type"].get("paraphrase", [])

                #     st.markdown("**Paraphrase candidates:**")
                #     if not raw_par:
                #         st.write("_No paraphrases generated._")
                #     else:
                #         for i, e in enumerate(raw_par, start=1):
                #             st.write(f"{i}. {e}")

                #     st.markdown("**Filtered paraphrases (kept):**")
                #     if not filt_par:
                #         st.write("_No paraphrases passed filtering thresholds._")
                #     else:
                #         for i, e in enumerate(filt_par, start=1):
                #             st.write(f"{i}. {e}")

                #     # Optional: show scores table
                #     if scores:
                #         st.markdown("**Scores per paraphrase (token overlap & cosine sim):**")
                #         st.dataframe(
                #             [
                #                 {
                #                     "text": s["text"],
                #                     "token_overlap": round(s["token_overlap"], 3),
                #                     "cos_sim": round(s["cos_sim"], 3),
                #                     "kept": s["kept"],
                #                 }
                #                 for s in scores
                #             ]
                #         )

                # # Show results side by side: baseline vs fused
                # # TILL HERE, INCLUDING THE LINE BELOW: 
                # st.subheader("Search Results")
                if use_expansion:
                    st.subheader("LLM-Generated Expansions")

                    expansions = results["expansions"]
                    filtered = results["filtered_expansions"]
                    scores_by_type = results["scores_by_type"]

                    # Show expansions for each type
                    for exp_type in ["paraphrase", "entity", "conceptual"]:
                        raw_list = expansions.get(exp_type, [])
                        filt_list = filtered.get(exp_type, [])
                        scores = scores_by_type.get(exp_type, [])

                        st.markdown(f"### {exp_type.capitalize()} expansions")

                        st.markdown("**Candidates:**")
                        if not raw_list:
                            st.write("_No candidates generated._")
                        else:
                            for i, e in enumerate(raw_list, start=1):
                                st.write(f"{i}. {e}")

                        st.markdown("**Kept after filtering:**")
                        if not filt_list:
                            st.write("_No expansions passed filtering._")
                        else:
                            for i, e in enumerate(filt_list, start=1):
                                st.write(f"{i}. {e}")

                        if scores:
                            st.markdown("**Scores (token overlap & cosine similarity):**")
                            st.dataframe(
                                [
                                    {
                                        "text": s["text"],
                                        "token_overlap": round(s["token_overlap"], 3),
                                        "cos_sim": round(s["cos_sim"], 3),
                                        "kept": s["kept"],
                                    }
                                    for s in scores
                                ]
                            )

                        st.markdown("---")

                st.subheader("Search Results")
                col_baseline, col_fused = st.columns(2)

                with col_baseline:
                    st.markdown("### Baseline BM25 (original query)")
                    for i, doc in enumerate(results["original"][:k_display], start=1):
                        st.markdown(f"**#{i}. {doc['title'] or '(no title)'}**")
                        st.caption(f"doc_id={doc['doc_id']} | score={doc.get('score', 0):.4f}")
                        snippet = doc["contents"][:400].replace("\n", " ")
                        st.write(snippet + ("..." if len(doc["contents"]) > 400 else ""))
                        st.markdown("---")

                with col_fused:
                    label = "BM25 + expansions + RRF" if use_expansion else "BM25 (same as baseline)"
                    st.markdown(f"### {label}")
                    for i, doc in enumerate(results["fused"][:k_display], start=1):
                        st.markdown(f"**#{i}. {doc['title'] or '(no title)'}**")
                        score = doc.get("fused_score", doc.get("score", 0.0))
                        st.caption(
                            f"doc_id={doc['doc_id']} | fused_score={score:.4f} | from={', '.join(doc.get('from_queries', []))}"
                        )
                        snippet = doc["contents"][:400].replace("\n", " ")
                        st.write(snippet + ("..." if len(doc["contents"]) > 400 else ""))
                        st.markdown("---")

    # ------------------------------------------------------------------
    # Tab 2: Case Studies
    # ------------------------------------------------------------------
    with tab2:
        st.header("Case Studies (Queries from Evaluation Set)")

        try:
            queries = load_queries()
            qrels = load_qrels()
        except FileNotFoundError as e:
            st.error(
                f"Could not load queries/qrels for case studies: {e}. "
                "Make sure data/queries.jsonl and data/qrels.jsonl exist."
            )
            queries = []
            qrels = {}

        if queries:
            qid_to_query = {q["qid"]: q["query"] for q in queries}
            selected_qid = st.selectbox(
                "Select a query:",
                options=list(qid_to_query.keys()),
                format_func=lambda qid: f"{qid}: {qid_to_query[qid]}",
            )

            k_cs = st.slider(
                "Top-k for this case study:",
                min_value=5,
                max_value=50,
                value=10,
                step=1,
                key="case_study_k",
            )

            if st.button("Run case study", type="primary"):
                q_text = qid_to_query[selected_qid]
                st.markdown(f"**Query ({selected_qid}):** {q_text}")

                # Baseline ranking
                base_docs = search_bm25(q_text, k=cfg["retrieval"].get("bm25_top_k", 50))

                # Expansions + fusion
                results = run_search_with_expansion(q_text, bm25_k=cfg["retrieval"].get("bm25_top_k", 50))

                # Metrics: baseline vs expansion for this query
                rel_lookup = qrels.get(selected_qid, {})
                relevant_doc_ids = [doc_id for doc_id, rel in rel_lookup.items() if rel > 0]

                base_top_k = base_docs[:k_cs]
                fused_top_k = results["fused"][:k_cs]

                base_metrics = compute_metrics_for_query(base_top_k, selected_qid, qrels, k_cs)
                exp_metrics = compute_metrics_for_query(fused_top_k, selected_qid, qrels, k_cs)

                st.subheader("Per-Query Metrics")

                st.table(
                    {
                        "": ["Precision", "Recall", "nDCG"],
                        "Baseline BM25": [
                            f"{base_metrics['precision_at_k']:.3f}",
                            f"{base_metrics['recall_at_k']:.3f}",
                            f"{base_metrics['ndcg_at_k']:.3f}",
                        ],
                        "BM25 + Expansions + RRF": [
                            f"{exp_metrics['precision_at_k']:.3f}",
                            f"{exp_metrics['recall_at_k']:.3f}",
                            f"{exp_metrics['ndcg_at_k']:.3f}",
                        ],
                    }
                )

                # Show rankings side-by-side
                st.subheader("Rankings")

                col_b, col_e = st.columns(2)
                with col_b:
                    st.markdown("### Baseline BM25")
                    for i, doc in enumerate(base_top_k, start=1):
                        st.markdown(f"**#{i}. {doc['title'] or '(no title)'}**")
                        st.caption(f"doc_id={doc['doc_id']} | score={doc.get('score', 0):.4f}")
                        snippet = doc["contents"][:300].replace("\n", " ")
                        st.write(snippet + ("..." if len(doc["contents"]) > 300 else ""))
                        st.markdown("---")

                with col_e:
                    st.markdown("### BM25 + expansions + RRF")
                    for i, doc in enumerate(fused_top_k, start=1):
                        st.markdown(f"**#{i}. {doc['title'] or '(no title)'}**")
                        score = doc.get("fused_score", doc.get("score", 0.0))
                        st.caption(
                            f"doc_id={doc['doc_id']} | fused_score={score:.4f} | from={', '.join(doc.get('from_queries', []))}"
                        )
                        snippet = doc["contents"][:300].replace("\n", " ")
                        st.write(snippet + ("..." if len(doc["contents"]) > 300 else ""))
                        st.markdown("---")

        else:
            st.info("No queries found for case studies. Please create data/queries.jsonl and data/qrels.jsonl.")

    # ------------------------------------------------------------------
    # Tab 3: Evaluation Summary
    # ------------------------------------------------------------------
    with tab3:
        st.header("Evaluation Summary (Baseline vs LLM-Augmented)")

        k_eval = st.slider(
            "k for evaluation metrics:",
            min_value=5,
            max_value=50,
            value=10,
            step=1,
            key="eval_k",
        )

        if st.button("Run full evaluation", type="primary"):
            with st.spinner("Running evaluation for baseline and LLM-augmented systems..."):
                agg_base, _ = cached_evaluate_baseline(k_eval)
                agg_exp, _ = cached_evaluate_with_expansion(k_eval)

            st.subheader("Aggregate Metrics")

            st.table(
                {
                    "": ["Precision", "Recall", "nDCG"],
                    "Baseline BM25": [
                        f"{agg_base['precision_at_k']:.3f}",
                        f"{agg_base['recall_at_k']:.3f}",
                        f"{agg_base['ndcg_at_k']:.3f}",
                    ],
                    "BM25 + Expansions + RRF": [
                        f"{agg_exp['precision_at_k']:.3f}",
                        f"{agg_exp['recall_at_k']:.3f}",
                        f"{agg_exp['ndcg_at_k']:.3f}",
                    ],
                }
            )

            st.markdown(
                """
                These metrics are computed over the query set in `data/queries.jsonl`
                with relevance labels from `data/qrels.jsonl`. This provides a
                quantitative comparison of the systems.
                """
            )


if __name__ == "__main__":
    main()
