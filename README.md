# Mini RAG-Based Academic Paper Assistant

A lightweight Retrieval-Augmented Generation (RAG) system that helps users discover relevant Deep Learning and Machine Learning research papers through natural language queries.

The project combines a fine-tuned Sentence Transformer model, FAISS vector search, and an LLM-based answer generation pipeline.

## Project Overview

Finding relevant academic papers through keyword-based search can be challenging, especially when users describe research problems using different terminology.

This project addresses that problem by building a semantic search engine that understands natural-language queries and retrieves relevant research papers based on their abstracts.

The system also uses Retrieval-Augmented Generation (RAG) to generate concise answers grounded in the retrieved papers.

## Architecture

The project consists of five main components:

1. **Dataset Preparation**
   - Source: `CShorten/ML-ArXiv-Papers`
   - 2,000 research papers selected.
   - Paper titles and abstracts used as document data.
   - Train/validation/test split: 1,500 / 200 / 300.

2. **Synthetic Query Generation**
   - OpenAI `gpt-4o-mini` generates one natural-language search query per paper.
   - Queries are generated using paper titles and abstracts.
   - The generated queries are designed to resemble real user search requests rather than simply copying paper titles.

3. **Contrastive Fine-Tuning**
   - Base model: `sentence-transformers/all-MiniLM-L6-v2`
   - Embedding dimension: 384
   - Framework: PyTorch and Sentence Transformers
   - Loss function: Multiple Negatives Ranking Loss
   - Training examples: 1,500 query-document pairs
   - Training duration: 1 epoch

4. **Semantic Search**
   - FAISS `IndexFlatIP` for similarity search.
   - Normalized embeddings for cosine similarity retrieval.
   - Top-K relevant papers retrieved for each query.

5. **RAG Answer Generation**
   - Retrieved paper abstracts are provided as context.
   - OpenAI `gpt-4o-mini` generates answers based on the retrieved information.
   - Relevant paper titles are included as citations.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Embedding Model | Sentence Transformers |
| Pretrained Model | all-MiniLM-L6-v2 |
| Fine-Tuning | Multiple Negatives Ranking Loss |
| Vector Search | FAISS |
| Dataset | Hugging Face Datasets |
| LLM | OpenAI gpt-4o-mini |
| Environment | Google Colab |

## Evaluation

The retrieval system was evaluated using 300 held-out test queries against the full collection of 2,000 paper abstracts.

Baseline and fine-tuned models were evaluated using the same queries and document collection.

### Baseline vs. Fine-Tuned Model

| Metric | Baseline | Fine-Tuned | Improvement |
|---|---:|---:|---:|
| Recall@1 | 91.00% | **93.33%** | +2.33 pp |
| Recall@5 | 98.33% | **99.00%** | +0.67 pp |
| MRR | 0.9419 | **0.9597** | +0.0177 |

Fine-tuning improved all three retrieval metrics.

The largest improvement was observed in Recall@1, indicating that the fine-tuned model was more successful at ranking the correct paper as the first result.

## Project Structure

```text
Mini_RAG_Based_Academic_Paper_Assistant/
│
├── Mini_RAG_Based_Academic_Paper_Assistant.ipynb
│
├── data/
│   ├── train_data.csv
│   ├── validation_data.csv
│   ├── test_data.csv
│   └── all_papers.csv
│
├── models/
│   ├── fine_tuned_minilm/
│   └── papers_faiss.index
│
└── README.md
```

## Example Usage

Example natural-language query:

```text
What deep learning approaches are used for
detecting diseases from medical images?
```

The system:
1. Converts the query into an embedding.
2. Retrieves the five most relevant papers using FAISS.
3. Builds a context from the retrieved abstracts.
4. Generates an answer using the LLM.
5. Returns the answer together with the retrieved sources.

## Limitations

- The dataset contains a selected subset of 2,000 papers rather than the entire arXiv collection.
- Search queries are synthetically generated using an LLM.
- Fine-tuning was performed for one epoch.
- Retrieval evaluation uses 300 held-out queries and a 2,000-document candidate collection.
- Generated answers depend on the relevance and completeness of retrieved abstracts.
- The system does not replace full-text academic paper analysis.

## Future Improvements

- Evaluate additional fine-tuning epochs and hyperparameters.
- Improve retrieval relevance through reranking.
- Introduce more diverse and human-written evaluation queries.
- Expand the research paper collection.
- Improve source-grounded answer generation.
- Add retrieval confidence and more detailed evaluation.

## Key Takeaway

This project demonstrates an end-to-end semantic retrieval and RAG pipeline, including domain-specific embedding fine-tuning, vector search, quantitative evaluation, and LLM-based answer generation.
