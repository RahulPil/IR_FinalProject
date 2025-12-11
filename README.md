# Information Retrieval with LLM-Based Query Expansion

This repository contains an end-to-end information retrieval (IR) system that compares a BM25 baseline with LLM-based query expansion. The system supports corpus construction, indexing, query expansion, LLM-generated relevance judgments (qrels), evaluation, and an interactive Streamlit interface.

---

## Requirements

- Python 3.10 or 3.11  
- Conda (or Miniconda)  
- An OpenAI API key

---

## 1. Clone the Repository

```bash
git clone https://github.com/RahulPil/IR_FinalProject.git
cd IR_FinalProject
```

---

## 2. Create and Activate Conda Environment

```bash
conda create -n ir_final python=3.11
conda activate ir_final
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Set OpenAI API Key

Set your OpenAI API key as an environment variable.

### macOS / Linux

```bash
export OPENAI_API_KEY="your_api_key_here"
```

### Windows (PowerShell)

```powershell
setx OPENAI_API_KEY "your_api_key_here"
```

---

## 5. Build the Dataset

Download, clean, and process a subset of Wikipedia into a JSONL corpus.

```bash
python -m scripts.prepare_wiki_subset \
    --output data/processed/wiki_subset.jsonl \
    --max-docs 50000
```

This script:
- Downloads Wikipedia articles
- Cleans and normalizes text
- Writes `(doc_id, title, text)` records to a JSONL file

---

## 6. Indexing (BM25)

Index the corpus using BM25.
```bash
python -m scripts.build_index \     
    --corpus data/processed/wiki_subset.jsonl \
    --output data/index/bm25_index.pkl
```

---

## 7. Create Queries

Queries are stored in `data/queries.jsonl`.

Each line must follow the format:

```json
{"qid": "q001", "query": "transformer architecture for deep learning"}
```

You may modify or replace this file with your own queries.

---

## 8. Generate Relevance Judgments (Qrels)
Generate temporary qrels.jsonl

```bash
python -m scripts.build_qrels_candidates \
    --corpus data/processed/wiki_subset.jsonl \
    --queries data/queries.jsonl \
    --output data/qrels_template.jsonl \
    --top-k 30
```

Use an LLM to automatically label relevance for each query–document pair.

```bash
python -m scripts.auto_label_qrels_with_llm \
    --corpus data/processed/wiki_subset.jsonl \
    --queries data/queries.jsonl \
    --output data/qrels.jsonl
```

This script:
- Retrieves candidate documents using BM25
- Assigns relevance labels {0, 1, 2} using the LLM
- Saves results to `data/qrels.jsonl`

---

## 9. Run Streamlit Application

Launch the interactive retrieval interface.

```bash
streamlit run app.py
```

The Streamlit app allows users to enter custom queries and compare BM25 results against expansion-augmented retrieval.

---

