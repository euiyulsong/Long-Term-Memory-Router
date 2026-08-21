# Long-Term Memory Retrieval + QA Ablation
## Metric Definitions

| Column                     | 의미                                                                                                                        |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **Query**                  | LTM 검색 query 구성 방식. `multiturn`은 현재 multi-turn context + question을 그대로 사용하고, `rewrite`는 LLM으로 standalone search query를 생성 |
| **Pre**                    | **Pre-Router** 사용 여부. Retrieval 전에 LTM 검색이 필요한지 `1/0`으로 판단                                                                |
| **Post**                   | **Post-Router** 사용 여부. Retrieval 후 각 memory가 질문에 relevant한지 `1/0`으로 판단하여 filtering                                        |
| **Overall EM**             | LoCoMo + SQuAD 전체의 Exact Match                                                                                            |
| **Overall F1**             | LoCoMo + SQuAD 전체의 token-level F1                                                                                         |
| **LoCoMo EM**              | **LTM이 필요한 LoCoMo** 데이터의 Exact Match                                                                                      |
| **LoCoMo F1**              | LTM이 필요한 LoCoMo 데이터의 token-level F1                                                                                       |
| **SQuAD EM**               | **LTM이 필요 없는 SQuAD** 데이터의 Exact Match                                                                                     |
| **SQuAD F1**               | LTM이 필요 없는 SQuAD 데이터의 token-level F1                                                                                      |
| **Recall@5**               | LoCoMo에서 **GT evidence memory가 retrieval Top-5에 하나 이상 포함된 비율**                                                            |
| **Post Filter Recall**     | LoCoMo에서 Post-Router까지 거친 후에도 **GT evidence가 남아 있는 비율**                                                                   |
| **Pre Router Recall**      | LTM이 필요한 LoCoMo에서 Pre-Router가 **검색해야 한다(`1`)고 판단한 비율**                                                                    |
| **Pre Router Specificity** | LTM이 필요 없는 SQuAD에서 Pre-Router가 **검색하지 말아야 한다(`0`)고 정확히 판단한 비율**                                                           |
| **Unnecessary Retrieval**  | SQuAD에서 불필요하게 LTM retrieval을 수행한 비율. **낮을수록 좋음**                                                                          |
| **SQuAD Post Empty Rate**  | 최종적으로 SQuAD의 LTM 결과가 **0개가 된 비율**. 높을수록 불필요한 memory가 잘 제거됨                                                                |
| **SQuAD Post Keep Rate**   | 검색된 SQuAD의 irrelevant memory 중 Post-Router 이후에도 **남아 있는 비율**. **낮을수록 좋음**                                                 |
| **Avg Raw Memories**       | Retrieval 직후 반환된 평균 memory 개수                                                                                             |
| **Avg Final Memories**     | Pre/Post routing을 모두 거친 후 최종적으로 answer generation에 전달된 평균 memory 개수                                                       |
| **LoCoMo Final Memories**  | LoCoMo에서 최종 answer generation에 전달된 평균 memory 개수                                                                           |
| **SQuAD Final Memories**   | SQuAD에서 최종 answer generation에 전달된 평균 memory 개수                                                                            |

### 핵심 지표

실험 의사결정에는 아래 지표를 중심으로 보면 된다.

| 목적                                 | 주요 Metric                     |  방향 |
| ---------------------------------- | ----------------------------- | :-: |
| Query Rewrite 효과                   | `Recall@5`, `LoCoMo F1`       |  ↑  |
| Pre-Router가 필요한 검색을 보존             | `Pre Router Recall`           |  ↑  |
| Pre-Router가 불필요한 검색을 차단            | `Pre Router Specificity`      |  ↑  |
| 불필요한 LTM 검색                        | `Unnecessary Retrieval`       |  ↓  |
| Post-Router가 GT evidence를 보존       | `Post Filter Recall`          |  ↑  |
| Post-Router가 irrelevant memory를 제거 | `SQuAD Post Empty Rate`       |  ↑  |
| 최종 QA 성능                           | `LoCoMo EM/F1`, `SQuAD EM/F1` |  ↑  |

## Results

| Query       |   Pre  |   Post  | LoCoMo EM | LoCoMo F1 | SQuAD EM |  SQuAD F1 | Recall@5 | Post Recall | Pre Recall | Pre Specificity | Unnecessary Retrieval | SQuAD Empty |
| ----------- | :----: | :-----: | --------: | --------: | -------: | --------: | -------: | ----------: | ---------: | --------------: | --------------------: | ----------: |
| Multi-turn  |   OFF  |   OFF   |      0.01 |     0.052 |     0.79 |     0.851 |     0.03 |        0.03 |       1.00 |            0.00 |                  1.00 |        0.00 |
| Multi-turn  |   OFF  |    ON   |      0.01 |     0.034 |     0.85 |     0.893 |     0.03 |        0.02 |       1.00 |            0.00 |                  1.00 |        0.98 |
| Multi-turn  |   ON   |   OFF   |      0.02 |     0.066 |     0.85 |     0.893 |     0.03 |        0.03 |       0.95 |            0.99 |                  0.01 |        0.99 |
| Multi-turn  |   ON   |    ON   |      0.01 |     0.034 |     0.85 |     0.893 |     0.03 |        0.02 |       0.95 |            0.99 |                  0.01 |        1.00 |
| **Rewrite** |   OFF  |   OFF   |      0.15 | **0.322** |     0.76 |     0.848 | **0.50** |    **0.50** |       1.00 |            0.00 |                  1.00 |        0.00 |
| Rewrite     |   OFF  |    ON   |  **0.18** |     0.276 | **0.85** | **0.893** | **0.50** |        0.44 |       1.00 |            0.00 |                  1.00 |        0.96 |
| **Rewrite** | **ON** | **OFF** |      0.15 | **0.322** | **0.85** | **0.893** |     0.48 |    **0.48** |   **0.95** |        **0.99** |              **0.01** |        0.99 |
| Rewrite     |   ON   |    ON   |  **0.18** |     0.276 | **0.85** | **0.893** |     0.48 |        0.42 |   **0.95** |        **0.99** |              **0.01** |    **1.00** |

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
