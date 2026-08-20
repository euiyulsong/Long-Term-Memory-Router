# Long-Term Memory Ablation Results

## Summary (Only 10)

| Query       |   Pre   |   Post  | Overall Judge | Needed Judge | Not-Needed Judge | Recall@5 | Post-filter Recall |
| ----------- | :-----: | :-----: | ------------: | -----------: | ---------------: | -------: | -----------------: |
| Multi-turn  |   OFF   |   OFF   |          0.40 |         0.00 |             0.80 |     0.00 |               0.00 |
| Multi-turn  |   OFF   |    ON   |          0.45 |         0.00 |             0.90 |     0.00 |               0.00 |
| Multi-turn  |    ON   |   OFF   |          0.45 |         0.00 |             0.90 |     0.00 |               0.00 |
| Multi-turn  |    ON   |    ON   |          0.45 |         0.00 |             0.90 |     0.00 |               0.00 |
| **Rewrite** | **OFF** | **OFF** |      **0.70** |     **0.50** |         **0.90** | **0.50** |           **0.50** |
| Rewrite     |   OFF   |    ON   |          0.60 |         0.30 |             0.90 |     0.50 |               0.30 |
| Rewrite     |    ON   |   OFF   |          0.45 |         0.00 |             0.90 |     0.00 |               0.00 |
| Rewrite     |    ON   |    ON   |          0.45 |         0.00 |             0.90 |     0.00 |               0.00 |

## Findings

### 1. Query Rewriting is Effective

```text
Multi-turn → Recall@5 = 0.00, Needed Judge = 0.00
Rewrite    → Recall@5 = 0.50, Needed Judge = 0.50
```

Query rewriting substantially improves LTM retrieval and downstream answer quality.

**Best configuration:** `Rewrite + No Pre-Router + No Post-Router`

---

### 2. Pre-Retrieval Router Hurts Performance

With the pre-router enabled:

```text
Pre-router Recall = 0.10
Pre-router Specificity = 1.00
```

The router successfully avoids unnecessary retrieval, but incorrectly blocks **90% of queries that actually require memory**.

As a result:

```text
Needed Judge: 0.50 → 0.00
Recall@5:     0.50 → 0.00
```

The current pre-router is therefore too conservative.

---

### 3. Post-Retrieval Self-Router Also Hurts

For the rewrite configuration:

```text
Without Post-Router:
Needed Judge       = 0.50
Post-filter Recall = 0.50

With Post-Router:
Needed Judge       = 0.30
Post-filter Recall = 0.30
```

The post-router reduces the average number of memories from:

```text
5.00 → 0.55
```

but also filters out relevant evidence.

Therefore, the current pointwise relevance filter is overly aggressive.

---

## Conclusion

The best pipeline in this experiment is:

```text
Multi-turn Conversation
        ↓
Query Rewrite
        ↓
LTM Retrieval Top-5
        ↓
Answer Generation
```

Current results suggest:

* **Query Rewrite:** Recommended
* **Pre-Retrieval Router:** Not recommended
* **Post-Retrieval Self-Router:** Not recommended

The main issue with both routers is **low recall on memory-required queries**, causing relevant evidence to be discarded before answer generation.
