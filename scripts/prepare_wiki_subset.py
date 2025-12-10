import json
from pathlib import Path

from datasets import load_dataset

from src.config import load_config, PROJECT_ROOT


def main():
    cfg = load_config()
    corpus_processed_path = PROJECT_ROOT / cfg["paths"]["corpus_processed_path"]
    corpus_processed_path.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading Wikimedia/Wikipedia dataset")

    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")

    num_docs = 50000
    print(f"[INFO] Sampling {num_docs} documents from the train split")
    dataset = dataset.shuffle(seed=42).select(range(num_docs))

    print(f"[INFO] Writing corpus to {corpus_processed_path}")
    with open(corpus_processed_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(dataset):
            title = row.get("title", "").strip()
            text = row.get("text", "").strip()
            if not text:
                continue

            doc = {
                "id": f"wiki_{i}",
                "title": title,
                "contents": text,
            }
            f.write(json.dumps(doc) + "\n")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
