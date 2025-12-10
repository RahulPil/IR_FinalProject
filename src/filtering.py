from typing import Dict, List, Tuple
from functools import lru_cache
import re

from sentence_transformers import SentenceTransformer
import numpy as np

from .config import load_config

_cfg = load_config()


@lru_cache(maxsize=1)
def _get_embedding_model() -> SentenceTransformer:
    model_name = _cfg["filtering"].get("embedding_model", "all-MiniLM-L6-v2")
    print(f"[INFO] Loading sentence-transformers model: {model_name}")
    return SentenceTransformer(model_name)


def _tokenize_for_overlap(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    return tokens


def _token_overlap(a: str, b: str) -> float:
    tokens_a = set(_tokenize_for_overlap(a))
    tokens_b = set(_tokenize_for_overlap(b))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / float(len(union))


def _embedding_similarity(a: str, b: str) -> float:
    model = _get_embedding_model()
    embeddings = model.encode([a, b], convert_to_numpy=True, normalize_embeddings=True)
    v1, v2 = embeddings[0], embeddings[1]
    return float(np.dot(v1, v2))


def score_expansion(original: str, expanded: str) -> Dict:
    overlap = _token_overlap(original, expanded)
    cos_sim = _embedding_similarity(original, expanded)
    return {
        "token_overlap": overlap,
        "cos_sim": cos_sim,
    }


def filter_expansions(
    original: str,
    expansions_by_type: Dict[str, List[str]],
    min_cos_sim: float = None,
    max_cos_sim: float = None,
    min_token_overlap: float = None,
) -> Tuple[Dict[str, List[str]], Dict[str, List[Dict]]]:
    if min_cos_sim is None:
        min_cos_sim = _cfg["filtering"].get("min_cos_sim", 0.4)
    if max_cos_sim is None:
        max_cos_sim = _cfg["filtering"].get("max_cos_sim", 0.98)
    if min_token_overlap is None:
        min_token_overlap = _cfg["filtering"].get("min_token_overlap", 0.1)

    exp_cfg = _cfg["expansion"]
    max_per_type = {
        "paraphrase": exp_cfg.get("max_paraphrase", 2),
        "entity": exp_cfg.get("max_entity", 2),
        "conceptual": exp_cfg.get("max_conceptual", 2),
    }

    alpha = 0.7  
    beta = 0.3   

    filtered: Dict[str, List[str]] = {}
    scores_by_type: Dict[str, List[Dict]] = {}

    for exp_type, exps in expansions_by_type.items():
        scored_list: List[Dict] = []

        for e in exps:
            metrics = score_expansion(original, e)
            overlap = metrics["token_overlap"]
            cos_sim = metrics["cos_sim"]
            eligible = True

            if cos_sim < min_cos_sim:
                eligible = False
            if cos_sim > max_cos_sim:
                eligible = False
            if overlap < min_token_overlap and cos_sim < (min_cos_sim + 0.1):
                eligible = False

            metrics.update(
                {
                    "text": e,
                    "eligible": eligible,
                    "kept": False,       
                    "rank_score": float("-inf"),  
                }
            )

            scored_list.append(metrics)

        for m in scored_list:
            if not m["eligible"]:
                continue
            m["rank_score"] = alpha * m["cos_sim"] + beta * m["token_overlap"]

        max_keep = max_per_type.get(exp_type, 2)
        kept_texts: List[str] = []

        if max_keep > 0:
            sorted_scored = sorted(scored_list, key=lambda x: x["rank_score"], reverse=True)
            kept_count = 0
            for m in sorted_scored:
                if kept_count >= max_keep:
                    break
                if not m["eligible"]:
                    continue
                m["kept"] = True
                kept_texts.append(m["text"])
                kept_count += 1

        filtered[exp_type] = kept_texts
        scores_by_type[exp_type] = scored_list

    return filtered, scores_by_type
