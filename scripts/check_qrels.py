import argparse
import json
from collections import Counter

QUERY_ID_FIELD = "qid"         
CORPUS_ID_FIELD = "id"              
QRELS_QUERY_ID_FIELD = "qid"   
QRELS_DOC_ID_FIELD = "doc_id"       
REL_FIELD = "relevance"             


def load_ids_from_jsonl(path, id_field):
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if id_field not in obj:
                raise ValueError(f"{id_field} not found in line from {path}: {obj}")
            ids.add(obj[id_field])
    return ids


def main(args):
    query_ids = load_ids_from_jsonl(args.queries, QUERY_ID_FIELD)
    num_queries = len(query_ids)

    doc_ids = load_ids_from_jsonl(args.corpus, CORPUS_ID_FIELD)
    num_docs = len(doc_ids)

    qrel_count = 0
    qrel_query_ids = set()
    bad_label_values = Counter()
    missing_query_ids = set()
    missing_doc_ids = set()

    with open(args.qrels, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qrel_count += 1

            qid = obj.get(QRELS_QUERY_ID_FIELD)
            did = obj.get(QRELS_DOC_ID_FIELD)
            rel = obj.get(REL_FIELD)

            if qid is None or did is None or rel is None:
                raise ValueError(f"Missing fields in qrels line: {obj}")

            qrel_query_ids.add(qid)

            if qid not in query_ids:
                missing_query_ids.add(qid)
            if did not in doc_ids:
                missing_doc_ids.add(did)

            try:
                rel_int = int(rel)
            except Exception:
                bad_label_values[rel] += 1
                continue
            if rel_int not in {0, 1, 2}:
                bad_label_values[rel_int] += 1

    print("=== Qrels Sanity Check ===")
    print(f"# queries in queries.jsonl        : {num_queries}")
    print(f"# distinct query_ids in qrels     : {len(qrel_query_ids)}")
    print(f"# documents in wiki_subset.jsonl  : {num_docs}")
    print(f"# total qrel lines                : {qrel_count}")

    if args.expected_top_k is not None and num_queries > 0:
        expected = num_queries * args.expected_top_k
        ratio = qrel_count / float(expected)
        print(f"Expected qrels (#queries * k)    : {expected}")
        print(f"qrels / expected ratio           : {ratio:.3f}")

    if missing_query_ids:
        print(f"[ERROR] {len(missing_query_ids)} qrel query_ids not in queries.jsonl")
    else:
        print("[OK] All qrel query_ids exist in queries.jsonl")

    if missing_doc_ids:
        print(f"[ERROR] {len(missing_doc_ids)} qrel doc_ids not in wiki_subset.jsonl")
    else:
        print("[OK] All qrel doc_ids exist in wiki_subset.jsonl")

    if bad_label_values:
        print("[ERROR] Found labels outside {0,1,2}:")
        for val, cnt in bad_label_values.items():
            print(f"  value={val!r} count={cnt}")
    else:
        print("[OK] All relevance labels in {0,1,2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, help="Path to queries.jsonl")
    parser.add_argument("--qrels", required=True, help="Path to qrels.jsonl")
    parser.add_argument("--corpus", required=True, help="Path to wiki_subset.jsonl")
    parser.add_argument(
        "--expected-top-k",
        type=int,
        default=None,
        help="Top-k used when labeling (for rough count check)",
    )
    args = parser.parse_args()
    main(args)
