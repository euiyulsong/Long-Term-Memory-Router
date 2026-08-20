# Long-Term Memory Retrieval Ablation

## Objective

This experiment evaluates three design choices for **Long-Term Memory (LTM) retrieval in multi-turn LLM systems**:

1. **Query Rewriting** — raw multi-turn context vs. standalone rewritten query
2. **Pre-Retrieval Router** — whether to retrieve LTM at all
3. **Post-Retrieval Self-Router** — whether to filter retrieved memories with a pointwise relevance check

## Dataset

We use **LoCoMo**, a benchmark for long-term conversational memory.

From eligible conversational QA examples, we sample **1,000 base questions** and create two matched conditions from each question:

| Condition             | Current Context                  | Long-Term Memory  | # Samples |
| --------------------- | -------------------------------- | ----------------- | --------: |
| **Memory Needed**     | Does not contain answer evidence | Contains evidence |     1,000 |
| **Memory Not Needed** | Contains answer evidence         | Evidence removed  |     1,000 |

**Total: 2,000 examples**

The question and gold answer are identical between each pair. Only the **location of the answer evidence** changes, allowing us to isolate the effect of LTM retrieval and routing.

```text
Same Question / Same Answer
           │
     ┌─────┴─────┐
     ▼           ▼
Memory Needed   Memory Not Needed
Evidence=LTM    Evidence=Current Context
```

## Pipeline

```text
Multi-turn Context + Query
          │
          ├── Query Rewrite (optional)
          ▼
    Retrieval Query
          │
          ├── Pre-Retrieval Router (optional)
          │       "Do we need LTM?"
          ▼
      LTM Retrieval
        Top-K = 5
          │
          ├── Pointwise Self-Router (optional)
          │       "Is this memory relevant?"
          ▼
     Answer Generation
```

## Models

* **LLM:** Qwen3-8B via OpenRouter
* **Retriever:** `BAAI/bge-base-en-v1.5`
* **Similarity:** Cosine similarity
* **Top-K:** 5
* **Temperature:** 0

Qwen3-8B is used for query rewriting, routing, relevance checking, and answer generation.

## Ablation

We evaluate all combinations of:

| Query      | Pre-Router | Post-Router |
| ---------- | :--------: | :---------: |
| Multi-turn |     OFF    |     OFF     |
| Multi-turn |     ON     |     OFF     |
| Multi-turn |     OFF    |      ON     |
| Multi-turn |     ON     |      ON     |
| Rewrite    |     OFF    |     OFF     |
| Rewrite    |     ON     |     OFF     |
| Rewrite    |     OFF    |      ON     |
| Rewrite    |     ON     |      ON     |

Each configuration is evaluated on all **2,000 examples**.

## Evaluation

### Query Rewriting

Compare:

```text
Multi-turn + No Router
vs.
Rewrite + No Router
```

Metrics: **Recall@5, QA Accuracy, F1**

### Pre-Retrieval Router

Compare:

```text
Rewrite + Pre-Router OFF
vs.
Rewrite + Pre-Router ON
```

Metrics:

* Memory-needed accuracy
* Memory-not-needed accuracy
* Router recall: `P(Retrieve | Memory Needed)`
* Unnecessary retrieval: `P(Retrieve | Memory Not Needed)`

Router recall is especially important because a false negative prevents the downstream retriever from accessing required evidence.

### Post-Retrieval Self-Router

Compare:

```text
Rewrite + Pre-Router
vs.
Rewrite + Pre-Router + Post-Router
```

Metrics:

* QA Accuracy / F1
* Evidence recall after filtering
* Average number of retained memories

## Main Questions

The experiment ultimately answers:

```text
1. Does query rewriting improve LTM retrieval over raw multi-turn context?

2. Can a pre-retrieval router avoid unnecessary LTM retrieval
   without hurting memory-dependent queries?

3. Does pointwise relevance filtering improve answer quality
   after retrieval?
```
