import argparse
import json
import os
import time
from pathlib import Path

import openai  

MODEL_NAME = "gpt-4o-mini"  
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 0.2  


def load_queries(queries_path: Path):
    qid_to_query = {}
    with queries_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid_to_query[obj["qid"]] = obj["query"]
    return qid_to_query


def load_corpus(corpus_path: Path):
    docid_to_doc = {}
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj["id"]
            title = obj.get("title", "")
            contents = obj.get("contents", "")
            docid_to_doc[doc_id] = {
                "title": title,
                "contents": contents,
            }
    return docid_to_doc


from openai import OpenAI
client = OpenAI()

def call_llm_for_relevance(query, title, contents):
    system_msg = (
        "You are an assistant that assigns relevance labels for an "
        "information retrieval evaluation.\n"
        "Given a query and a document, label:\n"
        "2 if the document is clearly about the query topic and very useful,\n"
        "1 if it is somewhat related or partially useful,\n"
        "0 if it is not really relevant.\n"
        "Respond with a single number: 0, 1, or 2. No explanation."
    )

    user_msg = (
        f"Query:\n{query}\n\n"
        f"Document title:\n{title}\n\n"
        f"Document contents (may be truncated):\n{contents[:2000]}"
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
            )
            content = response.choices[0].message.content.strip()

            # Extract first char in {0,1,2}
            for ch in content:
                if ch in ("0", "1", "2"):
                    return int(ch)

        except Exception as e:
            print(f"[WARN] LLM call failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            time.sleep(1.0)

    print("[WARN] LLM failed to produce a valid label, defaulting to 0.")
    return 0



def main():
    parser = argparse.ArgumentParser(
        description="Use GPT to assign relevance labels for qrels_template.jsonl."
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
        "--template",
        type=Path,
        default=Path("data/qrels_template.jsonl"),
        help="Path to qrels_template.jsonl (with relevance=0 placeholders).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/qrels.jsonl"),
        help="Path to final qrels.jsonl with GPT-labeled relevance.",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")

    print(f"[INFO] Loading queries from {args.queries} ...")
    qid_to_query = load_queries(args.queries)
    print(f"[INFO] Loaded {len(qid_to_query)} queries.")

    print(f"[INFO] Loading corpus from {args.corpus} ...")
    docid_to_doc = load_corpus(args.corpus)
    print(f"[INFO] Loaded {len(docid_to_doc)} documents.")

    print(f"[INFO] Reading template qrels from {args.template} ...")
    total = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.template.open("r", encoding="utf-8") as in_f, \
            args.output.open("w", encoding="utf-8") as out_f:

        for line in in_f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj["qid"]
            doc_id = obj["doc_id"]

            query = qid_to_query.get(qid)
            doc = docid_to_doc.get(doc_id)

            if query is None or doc is None:
                print(f"[WARN] Missing query or doc for qid={qid}, doc_id={doc_id}. Skipping.")
                continue

            title = doc["title"]
            contents = doc["contents"]

            relevance = call_llm_for_relevance(query, title, contents)

            record = {
                "qid": qid,           
                "doc_id": doc_id,     
                "relevance": relevance
            }
            out_f.write(json.dumps(record) + "\n")

            total += 1
            if total % 20 == 0:
                print(f"[INFO] Labeled {total} qrels so far...")
            time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"[INFO] Done. Wrote {total} labeled qrels to {args.output}")

def test():
    resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Say hello in one word."}
    ],
    )

    print(resp.choices[0].message.content)

if __name__ == "__main__":
    main()