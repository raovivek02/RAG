# MedLabel RAG

Grounded Q&A over FDA drug labels with citations and refusal.

**Educational project. Not for clinical use.** Do not use it to make medical decisions.

## How it works
1. Download 30 drug labels from openFDA (JSON, already split into sections).
2. Chunk each section by sentences (256 tokens, 40 overlap); every chunk starts with "Drug | Section".
3. Embed chunks with bge-small-en-v1.5 and index them in FAISS (exact inner-product search).
4. For a question, detect drug names and keep only those drugs' chunks, then take the top 20 by dense similarity.
5. Rerank the 20 with bge-reranker-base and fuse that ranking with the dense ranking (reciprocal rank fusion); keep the top 5.
6. Qwen2.5-7B-Instruct answers using only those passages, cites them as [n], and replies "I could not find this in the provided labels." when the answer is absent.

## Sample outputs

    Q: What is the maximum recommended daily dose of metformin for adults?
    A: The maximum recommended daily dose of metformin for adults is 2550 mg per day,
       given in divided doses [2, 3]. Doses above 2000 mg may be better tolerated
       when given 3 times a day with meals [2].

    Q: What is the recommended dose of ciprofloxacin for a urinary tract infection?
    A: I could not find this in the provided labels.     (ciprofloxacin is not in the 30 labels)

    Q: What is oral semaglutide indicated for?
    A: I could not find this in the provided labels.     (known failure: the Indications chunk was not retrieved)

## Setup
- Python 3.10. Dependencies: `pip install -r requirements.txt`. For an older NVIDIA driver, install a matching PyTorch build (I used `torch==2.6.0` from the cu124 index).
- Retrieval and reranking run on CPU. The 7B generator needs about 16 GB of GPU memory in bf16 (tested on an A100 40 GB).
- Without Slurm or a big GPU: pick a smaller model, for example

        MEDRAG_LLM=Qwen/Qwen2.5-1.5B-Instruct python src/rag.py "What are the contraindications of warfarin?"

  Answers from small models will be worse.
- With Slurm: `sbatch run_rag.sh` (demo) and `sbatch run_eval.sh` (RAG vs no-RAG evaluation).

## Reproduce
    python src/download_data.py && python src/fix_data.py
    python src/chunking.py --size 256 --overlap 40
    python src/build_index.py --size 256
    python src/eval_retrieval.py
    python src/eval_generation.py 256

## Results (25 hand-written questions: 21 answerable, 4 unanswerable)
Raw files: `results/retrieval.csv`, `results/generation.json`; questions and gold strings: `eval/qa.jsonl`.

| Retrieval (chunk 256) | recall@5 | MRR |
|---|---|---|
| dense | 0.95 | 0.93 |
| + rerank | 0.95 | 0.88 |
| + fusion | 0.95 | 0.93 |

| Generation | RAG | No retrieval |
|---|---|---|
| Answer accuracy (gold string match) | 0.95 | 0.62 |
| False refusals (answerable) | 1/21 | n/a |
| Correct refusals (unanswerable) | 4/4 | n/a |

With 21 answerable questions, one question is about 5 points, so differences of one or two questions are indicative only.

## Known failures
- Metformin: a pediatric limit (2000 mg) was given for an adult question; fixed by telling the model to name the patient group.
- Oral semaglutide: the Indications chunk was not retrieved (brand names RYBELSUS/OZEMPIC vs query wording).
- The reranker alone lowered MRR at 256 tokens; fusion removed the regression.
- Labels repeat text, so duplicate chunks fill the top 5 (not yet deduplicated).

## Next steps
Chunk deduplication, BM25 hybrid search, query-to-section routing, faithfulness checks, more drugs.
