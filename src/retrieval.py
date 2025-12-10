import pickle
from typing import List, Dict

from nltk.tokenize import word_tokenize

from .config import load_config, PROJECT_ROOT

_cfg = load_config()
_INDEX_DIR = (PROJECT_ROOT / _cfg["paths"]["index_dir"]).resolve()

_bm25 = None
_doc_metadata = None


def _load_index():
    global _bm25, _doc_metadata
    if _bm25 is not None and _doc_metadata is not None:
        return _bm25, _doc_metadata

    bm25_path = _INDEX_DIR / "bm25_index.pkl"
    meta_path = _INDEX_DIR / "doc_metadata.pkl"

    if not bm25_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"BM25 index files not found in {_INDEX_DIR}. "
            "Run scripts/build_index.py first."
        )

    with open(bm25_path, "rb") as f:
        _bm25 = pickle.load(f)

    with open(meta_path, "rb") as f:
        _doc_metadata = pickle.load(f)

    return _bm25, _doc_metadata


def _tokenize_query(query: str):
    return word_tokenize(query.lower())


def search_bm25(query: str, k: int = 20) -> List[Dict]:
    bm25, meta = _load_index()

    tokens = _tokenize_query(query)
    scores = bm25.get_scores(tokens)

    ranked_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:k]

    results = []
    for idx in ranked_indices:
        doc = meta[idx]
        results.append(
            {
                "doc_id": doc["doc_id"],
                "score": float(scores[idx]),
                "title": doc["title"],
                "contents": doc["contents"],
            }
        )

    return results
