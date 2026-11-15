# AskPolicy

Hybrid Retrieval and Domain-Adapted Generation for Question Answering over corporate legal documents (privacy policies, terms of service and product warranties)

Nobody reads these documents, and when you do need an answer for whether a document contains an answer to your specific interest, you end up scrolling through pages of legal text. AskPolicy takes a policy document and a plain-language question, finds the clause that answers it, and writes a short answer based only on that clause. If it can't find a relevant clause, it can say so instead of guessing.

This repository contains the code for the paper:

> Areen Abudayeh, Lama Shraim, Hamed Abdelhaq, Ahemd Alia. *AskPolicy: Hybrid Retrieval and Domain-Adapted Generation for Question Answering over Corporate Legal Documents.* 2026.
> [link / DOI to be added]

## How it works

The pipeline has three parts.

**1. Indexing.** The document (PDF or plain text) is split into chunks. Each chunk goes into two indexes: BM25 over stemmed tokens, and a FAISS index of `mxbai-embed-large-v1` embeddings. Indexes are built per document.

**2. Retrieval.** BM25 and FAISS each return their top chunks. The scores are normalised and combined with equal weights. The merged candidates are reranked with a cross-encoder (`BAAI/bge-reranker-large`). The top chunk is kept only if its reranker confidence and its cosine similarity scores to the question pass a defined threshold. Otherwise the generator receives the placeholder "No relevant clause was found."

**3. Generation.** Mistral 7B Instruct v0.2, fine-tuned with LoRA (4-bit, via Unsloth), reads the question and the chosen chunk in an Alpaca-style prompt and writes the answer.

## Results

All numbers are on a held-out test set of 165 questions from three documents the model never saw during fine-tuning (Netflix, TikTok, Snapchat).

| | Result |
|---|---|
| Retrieval, hybrid BM25 + FAISS | Precision@3 0.913, MRR 0.988, NDCG@3 0.986 |
| End-to-end, fine-tuned Mistral 7B | ROUGE-L 0.434, BLEU-2 0.351, Token F1 0.542 |
| End-to-end, GPT-4o-mini as generator (same retrieval) | ROUGE-L 0.249, BLEU-2 0.088, Token F1 0.324 |
| Without the cross-encoder reranker | ROUGE-L drops from 0.434 to 0.140 |

GPT-4o-mini does get a higher token recall (0.733 vs 0.576), because it writes longer answers. The fine-tuned model is better on every other metric we report. The paper has the full ablations: embedding models, retrieval strategies, other generators (LLaMA 3.1 8B, Qwen 2.5 7B), threshold sweeps, and an error analysis.

We also tried a variant that skips the generator entirely when retrieval fails and returns a fixed refusal. On a small set of 27 questions (15 unanswerable, 12 answerable) it refused all 15 unanswerable ones but also refused 3 of the 12 answerable ones. That set is tiny, so treat the result as a first signal and not a benchmark.

## Experiments

Every design choice in the pipeline was tested against alternatives. The paper reports all of them.

- Embedding model: `mxbai-embed-large-v1` vs. `all-MiniLM-L6-v2`
- Retrieval strategy: TF-IDF vs. BM25 vs. FAISS vs. hybrid BM25 + FAISS
- Cross-encoder reranking on and off
- Generator choice: Mistral 7B vs. LLaMA 3.1 8B vs. Qwen 2.5 7B, all fine-tuned with the same setup
- Fine-tuned Mistral vs. the base model vs. GPT-4o-mini, with retrieval held fixed
- 10-fold cross-validation of the fine-tuning
- Similarity threshold sweep
- Cross-encoder confidence threshold sweep
- Effect of adding retrieval-derived noisy samples to the training data
- Abstention on unanswerable questions, with and without gating the generator
- Qualitative error analysis, split by whether the failure came from retrieval or generation

(Noting that this repository only includes the final chosen framework code, no experimental code is included except the Abstention design.

## Repository contents

The main code lives in one notebook, `askpolicy_pipeline.ipynb`, which was written and run on Google Colab (single T4 GPU, about 14.5 GB).

| Section | What it does |
|---|---|
| 1–4 | Install dependencies, load Mistral 7B in 4-bit, attach LoRA, fine-tune with `SFTTrainer`, evaluate and plot loss curves |
| 5–6 | Save the adapter and tokenizer, then reload them for inference |
| 7 | Hybrid retrieval, reranking, filtering and answer generation |
| 8 | Retrieval evaluation (Precision/Recall/F1@k, MAP, MRR, NDCG) |
| 9 | End-to-end evaluation (BLEU, ROUGE-L, semantic agreement, token overlap) |
| 10 | Abstention experiment on answerable vs. unanswerable questions |
| 11 | Small Gradio demo: upload a document, ask questions |

## Running it

You need a GPU with roughly 15 GB of memory. The notebook assumes Colab, so a few paths (`/content/...`, Google Drive) are Colab-specific and will need changing if you run it elsewhere.

Expected inputs:

- `/content/train_data.json` is the fine-tuning set, a JSON list of `{"instruction", "input", "output"}` records (Dataset-A, 300 samples).
- `/content/tests/` is the evaluation set, with one JSON file and one matching PDF per document (Dataset-B). The JSON holds the question–answer records and the PDF is the document they refer to.
- `/content/no_answer/` is only needed for Section 10. It holds one JSON file per test document, named after it, with the unanswerable questions. Their `input` field is `"No relevant clause was found."`.

Then run the notebook top to bottom. Section 6 reloads the saved adapter from Google Drive (`models-askpolicy/`), so if you skip training you'll need to point those three zip paths at your own copies. Sections 8–10 need Sections 6 and 7 to have run first.


## Data

The dataset has 465 question–answer pairs built from ten public documents: privacy policies from Meta, TikTok and Snapchat; terms from eBay, Nike, SHEIN and Netflix; and warranties from Amazon, Apple and Samsung. We collected them in May 2025, and the live versions have probably changed since. 

The source documents belong to their respective companies. We link to them in the paper and don't claim any rights over them.

AskPolicy was built as a research project, and we hope it’s a useful starting point for anyone working on question answering over legal documents.
