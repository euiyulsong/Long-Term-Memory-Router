# Long-Term Memory Retrieval Ablation Results

## Results

| Query       |   Pre   |   Post  | Recall@5 | Post Recall | Pre Recall | Pre Specificity | SQuAD Empty |
| ----------- | :-----: | :-----: | -------: | ----------: | ---------: | --------------: | ----------: |
| Multi-turn  |   OFF   |   OFF   |     0.03 |        0.03 |       1.00 |            0.00 |        0.00 |
| Multi-turn  |   OFF   |    ON   |     0.03 |        0.02 |       1.00 |            0.00 |        0.98 |
| Multi-turn  |    ON   |   OFF   |     0.03 |        0.03 |       0.95 |            0.99 |        0.99 |
| Multi-turn  |    ON   |    ON   |     0.03 |        0.02 |       0.95 |            0.99 |        1.00 |
| **Rewrite** | **OFF** | **OFF** | **0.49** |    **0.49** |       1.00 |            0.00 |        0.00 |
| Rewrite     |   OFF   |    ON   |     0.49 |        0.43 |       1.00 |            0.00 |        0.96 |
| **Rewrite** |  **ON** | **OFF** | **0.47** |    **0.47** |   **0.95** |        **0.99** |    **0.99** |
| Rewrite     |    ON   |    ON   |     0.47 |        0.41 |       0.95 |            0.99 |        1.00 |

## Key Findings

### 1. Query Rewrite: Strongly Recommended

```text
Multi-turn Recall@5 : 0.03
Rewrite Recall@5    : 0.49
```

Query rewriting improves retrieval recall by **+46%p**. Raw multi-turn context is highly ineffective as a direct embedding query.

### 2. Pre-Router: Effective

```text
Memory-needed Recall : 0.95
SQuAD Specificity    : 0.99
Unnecessary Retrieval: 0.01
```

The pre-router correctly allows **95%** of necessary retrievals while blocking **99%** of unnecessary SQuAD retrievals.

With rewrite, Recall@5 decreases only slightly:

```text
0.49 → 0.47
```

while unnecessary retrieval drops:

```text
1.00 → 0.01
```

Therefore, the **pre-router provides a strong efficiency/quality trade-off**.

### 3. Post-Router: Not Recommended

With rewrite:

```text
Recall@5           : 0.49
Post-filter Recall : 0.43
```

The post-router successfully removes irrelevant SQuAD memories (`96–100%` empty), but also removes relevant LoCoMo evidence.

With Pre + Post:

```text
0.47 → 0.41
```

Thus, the pointwise post-filter introduces unnecessary recall loss.

## Conclusion

Recommended pipeline:

```text
Multi-turn Conversation
        ↓
Pre-Retrieval Router
        ↓
Query Rewrite
        ↓
Top-5 LTM Retrieval
```

* **Query Rewrite:** ✅ Strongly beneficial
* **Pre-Router:** ✅ Recommended
* **Post-Router:** ❌ Not recommended

The best practical configuration is **`Rewrite + Pre-Router + No Post-Router`**, retaining `0.47 Recall@5` while reducing unnecessary retrieval from `100%` to `1%`.
