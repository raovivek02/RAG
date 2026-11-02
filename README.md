# MedLabel RAG: grounded Q&A over FDA drug labels

Answers questions about 30 FDA drug labels (openFDA) with citations, and refuses when the labels don't contain the answer.

## Pipeline
openFDA JSON -> section-aware chunking (256 tokens, 40 overlap; header "Drug | Section" on every chunk)
-> bge-small-en-v1.5 embeddings -> FAISS exact index
-> drug-name filter -> top-20 dense -> bge-reranker-base
-> reciprocal-rank fusion (dense + reranker) -> top-5
-> Qwen2.5-7B-Instruct, prompt: use only numbered passages, cite [n], refuse if absent, state patient group.

## Run
    conda activate medrag
    python src/download_data.py && python src/fix_data.py
    python src/chunking.py --size 256 --overlap 40
    python src/build_index.py --size 256
    python src/eval_retrieval.py
    sbatch run_eval.sh        # GPU job (A100): RAG vs no-RAG, writes results/generation.json
    sbatch run_rag.sh         # demo questions

## Results (25 hand-written questions: 21 answerable, 4 unanswerable)
Retrieval, recall@5 / MRR: dense 0.95 / 0.93; +rerank 0.95 / 0.88; +fusion 0.95 / 0.93 (chunk 256).
Generation, answer accuracy: RAG 0.95 vs no-RAG 0.62. False refusals 1/21.
n is small (one question is about 5 points), so differences of one or two questions are indicative only.

## Known failures
- Metformin: pediatric limit (2000 mg) was returned for an adult question. Fixed by telling the model to name the patient group.
- Oral semaglutide: the Indications chunk was not retrieved (brand names RYBELSUS/OZEMPIC vs query wording).
- Reranker alone lowered MRR at 256 tokens; fusion removed the regression.
- Labels repeat text (e.g. levothyroxine), so duplicate chunks fill the top-5. Not yet deduplicated.

## Next steps
BM25 hybrid search, query-to-section routing, chunk deduplication, LLM-judge faithfulness, more drugs.

Not medical advice.
