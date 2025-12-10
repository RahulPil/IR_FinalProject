import json
import pickle
from pathlib import Path
from typing import List, Dict

from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

from .config import load_config, PROJECT_ROOT


def load_corpus() -> List[Dict]:
    """
    Load the processed Wikipedia subset from a JSONL file.
    Each line: {"id": ..., "title": ..., "contents": ...}
    """
    cfg = load_config()
    corpus_path = PROJECT_ROOT / cfg["paths"]["corpus_processed_path"]

    docs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def _tokenize(text: str) -> List[str]:
    return word_tokenize(text.lower())


def build_bm25_index() -> None:
    """
    Build a BM25 index over the JSONL corpus using rank_bm25 (pure Python),
    and save the BM25 object + document metadata to disk.
    """
    cfg = load_config()
    corpus_path = PROJECT_ROOT / cfg["paths"]["corpus_processed_path"]
    index_dir = PROJECT_ROOT / cfg["paths"]["index_dir"]

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus file not found at {corpus_path}. "
            "Run scripts/prepare_wiki_subset.py first."
        )

    index_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading corpus...")
    docs = load_corpus()
    print(f"[INFO] Loaded {len(docs)} documents.")

    print("[INFO] Tokenizing documents...")
    tokenized_docs = []
    doc_metadata = [] 
    for doc in docs:
        doc_id = doc.get("id")
        title = doc.get("title", "")
        contents = doc.get("contents", "")

        tokens = _tokenize(contents)
        tokenized_docs.append(tokens)
        doc_metadata.append(
            {
                "doc_id": doc_id,
                "title": title,
                "contents": contents,
            }
        )

    print("[INFO] Building BM25 index in memory...")
    bm25 = BM25Okapi(tokenized_docs)

    bm25_path = index_dir / "bm25_index.pkl"
    meta_path = index_dir / "doc_metadata.pkl"

    print(f"[INFO] Saving BM25 index to {bm25_path} ...")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)

    print(f"[INFO] Saving document metadata to {meta_path} ...")
    with open(meta_path, "wb") as f:
        pickle.dump(doc_metadata, f)

    print("[INFO] Done building BM25 index.")
