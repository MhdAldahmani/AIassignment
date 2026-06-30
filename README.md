# Disneyland Reviews — AI Assignment

LLM-powered analysis of 42,636 customer reviews across three Disneyland parks (California, Paris, Hong Kong).

## Pipeline

```
Raw CSV → Clean → LLM Enrichment → Insights & Q&A Endpoint
```

1. **Clean** — fix encoding, remove duplicates, parse dates into year/month/season.
2. **Enrich** — run every review through an LLM (Groq / `openai/gpt-oss-20b`) to tag 9 aspects (queues, staff, price, food, rides, cleanliness, crowding, weather, accessibility) with per-aspect sentiment and a short complaint/delight phrase. Results are cached to disk so this one-time step never reruns.
3. **Index** — embed all reviews locally with `BAAI/bge-m3` (MPS GPU) and store in Chroma with the enriched metadata as filterable fields.
4. **Ask** — a natural-language question becomes a structured filter (branch, country, season, aspect tags…) extracted by an LLM, applied to the DataFrame for real aggregate stats, and to Chroma for representative quoted snippets. A final LLM call synthesizes the answer, grounded only in that computed evidence.

## Notebooks

| Notebook | What it shows |
|---|---|
| `01_analysis.ipynb` | Data cleaning summary + 6 text-grounded, actionable CX insights (e.g. Paris staff 51% negative, California Winter worst for crowds, "friendly staff" is both the #1 delight driver and #2 complaint source) |
| `02_rag_endpoint.ipynb` | How the Q&A pipeline works, a worked example, 25-question evaluation (100% filter accuracy, 5.0/5 relevance, 3.32/5 faithfulness), and a free-form question cell to try your own questions |

