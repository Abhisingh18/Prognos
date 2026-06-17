# AI Intern Assignment — Submission

All four questions have been completed. The top three scores will be considered.

## Files in this submission

| File | Question | Description |
|------|----------|-------------|
| `Q1_Q2_conceptual_answers.md` | Q1, Q2 | Appointment-scheduling chatbot (challenges, evaluations, hallucination handling) and the full RAG FAQ-chatbot design with RAG / Advanced RAG / GraphRAG comparison. |
| `Q3_langgraph_blog_agent.py` | Q3 | LangGraph agent that reads topics from Excel, writes a blog per topic, saves each to a Google Doc, and writes the link back to the sheet. Runs as a self-contained simulation; the real LangGraph, Google Docs, and Sheets calls are provided as commented implementations behind clean function boundaries. |
| `Q4_surge_prediction.py` | Q4 | End-to-end demand-surge predictor. Generates synthetic data matching all five input schemas, builds the leakage-safe `surge_flag` label, engineers features (sales trend, stock pressure, disaster proximity, store/SKU metadata), trains a gradient-boosted classifier, and reports the appropriate metrics. |

## How to run the code

```
python Q3_langgraph_blog_agent.py     # prints the per-topic processing loop
python Q4_surge_prediction.py         # trains the model and prints metrics
```

Q4 requires `pandas`, `numpy`, and `scikit-learn`.

## Q3 — evaluating blog quality at scale (without reading each one manually)

1. **LLM-as-judge with a fixed rubric.** Each blog is scored (1–5) on topic relevance, structure, factual grounding, and readability via a separate model call. Anything below a set threshold is flagged for human review.
2. **Automatic checks.** Word-count band, presence of headings, a readability score (e.g. Flesch), topic-keyword coverage, and near-duplicate detection across posts.
3. **Groundedness check.** Where source material exists, verify that the blog's claims are supported by it.
4. **Sampled human audit.** A random 5–10% is read by a person to calibrate and validate the automated judge.

## Q4 — key modelling decisions

- **Time-based split**, not random, because this is a forecasting problem — the model trains on earlier dates and is tested on later ones.
- **PR-AUC is reported** alongside ROC-AUC, because surges are rare (around 4% of rows) and plain accuracy would be misleading.
- **No data leakage:** all features use information available up to the prediction date, while the label is computed from the following seven days. The final incomplete window per series is dropped.
- **Threshold tuning** is left to the retailer's cost trade-off: a missed surge (a stockout during a disaster) is far more costly than over-stocking, so the operating point should favour recall on the positive class.

## Assumptions

Each code file documents its own assumptions in a header comment. The main ones: the Excel sheet uses the columns shown in the prompt and rows already containing a blog link are skipped (Q3); the seven-day surge horizon is made explicit as `demand_next_7_days > 1.5 × 7 × daily_baseline` (Q4); and the disaster feed is joined to stores by haversine distance and alert recency (Q4).
