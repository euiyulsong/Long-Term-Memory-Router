# Long-Term Memory Retrieval + QA Ablation

## Results

| Query       |   Pre  |   Post  | LoCoMo EM | LoCoMo F1 | SQuAD EM |  SQuAD F1 | Recall@5 | Post Recall | Pre Recall | Pre Specificity |
| ----------- | :----: | :-----: | --------: | --------: | -------: | --------: | -------: | ----------: | ---------: | --------------: |
| Multi-turn  |   OFF  |   OFF   |      0.01 |     0.052 |     0.79 |     0.851 |     0.03 |        0.03 |       1.00 |            0.00 |
| Multi-turn  |   OFF  |    ON   |      0.01 |     0.034 |     0.85 |     0.893 |     0.03 |        0.02 |       1.00 |            0.00 |
| Multi-turn  |   ON   |   OFF   |      0.02 |     0.066 |     0.85 |     0.893 |     0.03 |        0.03 |       0.95 |            0.99 |
| Multi-turn  |   ON   |    ON   |      0.01 |     0.034 |     0.85 |     0.893 |     0.03 |        0.02 |       0.95 |            0.99 |
| **Rewrite** |   OFF  |   OFF   |  **0.15** | **0.322** |     0.76 |     0.848 | **0.50** |    **0.50** |       1.00 |            0.00 |
| Rewrite     |   OFF  |    ON   |  **0.18** |     0.276 | **0.85** | **0.893** | **0.50** |        0.44 |       1.00 |            0.00 |
| **Rewrite** | **ON** | **OFF** |      0.15 | **0.322** | **0.85** | **0.893** |     0.48 |    **0.48** |   **0.95** |        **0.99** |
| Rewrite     |   ON   |    ON   |  **0.18** |     0.276 | **0.85** | **0.893** |     0.48 |        0.42 |   **0.95** |        **0.99** |

## Key Findings

### 1. Query Rewrite — Strongly Recommended

Retrieval 성능이 크게 향상됐다.

```text
Recall@5
Multi-turn : 0.03
Rewrite    : 0.50
```

LoCoMo QA 성능도 크게 개선됐다.

```text
LoCoMo F1
Multi-turn : 0.052
Rewrite    : 0.322
```

→ **Multi-turn 전체를 retrieval query로 사용하는 것보다 standalone query rewrite가 명확하게 우수하다.**

---

### 2. Pre-Router — Recommended

Pre-router 성능:

```text
Recall      = 0.95
Specificity = 0.99
```

즉,

* 필요한 LTM retrieval의 **95%를 통과**
* 필요 없는 retrieval의 **99%를 차단**

Rewrite 기준 retrieval 손실도 작다.

```text
Recall@5
Pre OFF : 0.50
Pre ON  : 0.48
```

반면 unnecessary retrieval은:

```text
1.00 → 0.01
```

SQuAD EM도 `0.76 → 0.85`로 개선된다.

→ **2%p의 Recall 손실로 불필요한 retrieval을 거의 완전히 제거하므로 Pre-router 사용이 유리하다.**

---

### 3. Post-Router — Trade-off

Post-router는 불필요한 memory를 매우 잘 제거한다.

```text
SQuAD Post Empty Rate
Pre OFF + Post ON : 0.96
Pre ON  + Post ON : 1.00
```

하지만 relevant evidence도 일부 제거한다.

```text
Rewrite + Pre

Post OFF : Recall = 0.48
Post ON  : Recall = 0.42
```

LoCoMo에서는:

```text
             EM      F1
Post OFF    0.15    0.322
Post ON     0.18    0.276
```

→ EM은 소폭 상승하지만 **F1과 retrieval recall은 하락**한다. 따라서 Post-router가 완전히 나쁜 것은 아니지만, 이미 좋은 Pre-router가 있다면 추가적인 이득은 제한적이다.

## Conclusion

추천 pipeline:

```text
Multi-turn Conversation
        ↓
Pre-Router
        ↓
Query Rewrite
        ↓
LTM Retrieval Top-5
        ↓
Answer Generation
```

**최종 판단**

* **Query Rewrite:** ✅ 사용
* **Pre-Router:** ✅ 사용
* **Post-Router:** ⚠️ Optional / 기본적으로 제외

가장 균형 잡힌 설정은 **`Rewrite + Pre-Router + No Post-Router`**다. `Recall@5 = 0.48`을 유지하면서 불필요한 retrieval을 `1%`까지 낮추고, LoCoMo F1도 가장 높은 `0.322`를 유지한다.
