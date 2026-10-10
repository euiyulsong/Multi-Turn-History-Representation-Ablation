# Multi-turn History Representation Ablation

## 1. 실험 개요

GPT-6 Luna를 사용하여 멀티턴 QA에서 이전 대화 기록을 전달하는 세 가지 전략을 비교했다.

| 전략                | 설명                                |
| ----------------- | --------------------------------- |
| Raw Multi-turn    | 전체 이전 Q/A와 현재 질문을 그대로 전달          |
| History Selection | 현재 질문에 필요한 과거 Q/A만 선택하여 전달        |
| One-line Rewrite  | 이전 문맥을 반영해 현재 질문을 독립적인 단일 질문으로 변환 |

### 실험 환경

- **Model:** GPT-6 Luna
- **데이터셋:** CoQA (Easy), ConvFinQA (Hard)
- **평가 샘플:** 데이터셋별 100개, 총 200개
- **전략별 평가:** 각 200개, 총 600개 답변 생성
- **Context:** 모든 전략에 동일한 원본 근거 문서 제공
- **History:** 이전 턴의 Gold Answer 사용
- **Error:** 모든 전략에서 0건
- **평가지표:** EM, Token F1, Numeric Accuracy, Latency, Token Usage

본 실험은 검색 성능을 배제하고, 대화 히스토리 표현 방식이 최종 QA 성능에 미치는 영향을 분석한다.

---

## 2. 정확도 비교

### 2.1 주요 결과

| Dataset   | Strategy           |  Accuracy |  Token F1 | Accuracy Δ vs Raw |
| --------- | ------------------ | --------: | --------: | ----------------: |
| CoQA      | **Raw Multi-turn** | **0.320** | **0.559** |                 — |
| CoQA      | History Selection  |     0.260 |     0.489 |              -6%p |
| CoQA      | One-line Rewrite   |     0.290 |     0.500 |              -3%p |
| ConvFinQA | **Raw Multi-turn** | **0.770** |     0.362 |                 — |
| ConvFinQA | History Selection  |     0.730 |     0.343 |              -4%p |
| ConvFinQA | One-line Rewrite   |     0.760 | **0.366** |              -1%p |

### 2.2 CoQA (Easy)

- Raw Multi-turn이 EM 32%, F1 0.559로 가장 높았다.
- History Selection은 EM이 6%p, F1이 0.070 감소했다.
- One-line Rewrite는 EM이 3%p, F1이 0.059 감소했다.
- 전체 히스토리를 유지하는 방식이 두 가지 전처리 방식보다 좋은 결과를 보였다.

**해석:** CoQA에서는 이전 대화의 참조 관계를 그대로 유지하는 것이 유리했다. Selection 과정에서 필요한 히스토리가 누락되거나, Rewrite 과정에서 일부 문맥 정보가 소실됐을 가능성이 있다. 정확한 원인은 개별 모델 출력을 확인해야 한다.

### 2.3 ConvFinQA (Hard)

- Raw Multi-turn의 Numeric Accuracy가 77%로 가장 높았다.
- History Selection은 73%로 4%p 감소했다.
- One-line Rewrite는 76%로 Raw 대비 1%p 낮았다.
- Token F1은 Rewrite가 0.366으로 가장 높았지만 차이는 매우 작다.

**해석:** 금융 문서 기반 멀티턴 수치 추론에서도 Raw Multi-turn이 가장 높은 정확도를 기록했다. Rewrite는 Raw와 근접한 성능을 보인 반면, Selection은 상대적으로 큰 성능 하락을 보였다.

---

## 3. Token 및 Latency 비교

### 3.1 전체 비용

| Dataset   | Strategy           | Answer Input Tokens | Total Tokens | Answer Latency | Total Latency |
| --------- | ------------------ | ------------------: | -----------: | -------------: | ------------: |
| CoQA      | **Raw Multi-turn** |               522.8 |    **557.5** |          1.54s |     **1.54s** |
| CoQA      | History Selection  |               466.1 |        905.4 |          1.40s |         4.15s |
| CoQA      | One-line Rewrite   |               445.0 |        765.0 |          1.34s |         3.67s |
| ConvFinQA | **Raw Multi-turn** |              1066.4 |   **1182.4** |          2.06s |     **2.06s** |
| ConvFinQA | History Selection  |              1040.6 |       1545.7 |          2.03s |         4.32s |
| ConvFinQA | One-line Rewrite   |              1002.8 |       1388.0 |          1.83s |         3.75s |

Total Tokens와 Total Latency에는 History Selection 또는 Rewrite의 전처리 LLM 호출 비용이 포함된다.

### 3.2 Raw 대비 비용 변화

| Dataset   | Strategy          | Total Token 증가 | Total Latency 증가 |
| --------- | ----------------- | -------------: | ---------------: |
| CoQA      | History Selection |         +62.4% |          +169.5% |
| CoQA      | One-line Rewrite  |         +37.2% |          +138.3% |
| ConvFinQA | History Selection |         +30.7% |          +109.7% |
| ConvFinQA | One-line Rewrite  |         +17.4% |           +82.0% |

**주요 관찰**

1. History Selection과 Rewrite 모두 최종 답변 단계의 Input Token을 줄였다.
2. 별도의 LLM 전처리 호출로 인해 전체 Token 사용량과 Latency는 오히려 증가했다.
3. Raw Multi-turn은 두 데이터셋에서 가장 낮은 전체 비용을 보였다.
4. 본 실험의 평균 히스토리 길이에서는 전처리에 따른 토큰 절감이 추가 호출 비용을 상쇄하지 못했다.

---

## 4. History Selection 분석

| Dataset   | Raw 평균 턴 수 | Selection 평균 턴 수 | 히스토리 감소 |
| --------- | ---------: | ---------------: | ------: |
| CoQA      |       4.43 |             1.62 |   63.4% |
| ConvFinQA |       3.26 |             2.21 |   32.2% |

History Selection은 실제로 답변 모델에 전달되는 Q/A 수를 줄였다.

그러나 다음과 같은 결과가 나타났다.

- CoQA: 히스토리 63.4% 감소, EM 6%p 하락
- ConvFinQA: 히스토리 32.2% 감소, Numeric Accuracy 4%p 하락

**해석:** 히스토리 압축은 성공했지만 정확도 개선으로 이어지지 않았다. 일부 이전 턴이 정답 추론에 필요한 정보를 포함했을 가능성이 있으며, 현재 길이에서는 전체 히스토리 처리 비용도 상대적으로 낮았다.

---

## 5. Paired Win / Tie / Loss

동일한 질문 100개에서 두 전략의 정답 여부를 직접 비교했다.

| Dataset   | Comparison           | Win | Tie | Loss | Δ Accuracy | 95% CI      |
| --------- | -------------------- | --: | --: | ---: | ---------: | ----------- |
| CoQA      | Selection vs Raw     |   2 |  90 |    8 |       -6%p | [-12%, 0%]  |
| CoQA      | Rewrite vs Raw       |   2 |  93 |    5 |       -3%p | [-8%, +2%]  |
| CoQA      | Selection vs Rewrite |   2 |  93 |    5 |       -3%p | [-8%, +2%]  |
| ConvFinQA | Selection vs Raw     |   5 |  86 |    9 |       -4%p | [-11%, +3%] |
| ConvFinQA | Rewrite vs Raw       |   7 |  85 |    8 |       -1%p | [-8%, +6%]  |
| ConvFinQA | Selection vs Rewrite |   6 |  85 |    9 |       -3%p | [-11%, +4%] |

- **Win:** 첫 번째 전략만 정답
- **Tie:** 두 전략 모두 정답이거나 모두 오답
- **Loss:** 두 번째 전략만 정답
- **CI:** Paired Bootstrap, 2,000회 재표본추출

### 통계적 해석

- CoQA에서 History Selection은 Raw 대비 8개 질문에서 패배하고 2개에서 승리했다.
- CoQA Rewrite는 Raw 대비 5개 질문에서 패배하고 2개에서 승리했다.
- ConvFinQA Rewrite는 Raw 대비 7승 8패로 거의 동일한 정답률을 보였다.
- 모든 95% Bootstrap CI가 0을 포함하므로, 정확도 차이가 통계적으로 명확하다고 보기 어렵다.

현재 결과는 Raw Multi-turn의 우세 경향을 보여주지만, 정확도의 일반적인 우위를 확정하는 근거로는 제한적이다.

---

## 6. 주요 실패 및 성공 사례

### 6.1 CoQA

| 질문                                         | Gold            | Raw | Selection | Rewrite |
| ------------------------------------------ | --------------- | --- | --------- | ------- |
| What was the official name?                | Deutsches Reich | ✅   | ❌         | ✅       |
| Where was that?                            | Los Angeles     | ❌   | ✅         | ❌       |
| Did they look alike?                       | Yes             | ✅   | ❌         | ❌       |
| Where does Nicole live?                    | Shanghai        | ✅   | ✅         | ❌       |
| Whose throw was behind Tom's by 14 inches? | Larson          | ✅   | ❌         | ❌       |

**관찰:** Raw가 맞히고 전처리 전략이 틀리는 사례가 더 많았지만, History Selection만 정답인 사례도 존재했다.

이 사례들만으로는 문맥 누락, 질의 의미 변경 또는 답변 형식 오류 중 어떤 요인이 성능 차이를 만들었는지 확정할 수 없다.

### 6.2 ConvFinQA

| 사례 ID                          | Gold          | Raw | Selection | Rewrite |
| ------------------------------ | ------------- | --- | --------- | ------- |
| Double_AMT/2010/page_34.pdf:5  | 22556579533.3 | ❌   | ✅         | ✅       |
| Double_C/2008/page_102.pdf:3   | 108.4         | ✅   | ❌         | ❌       |
| Double_CDNS/2006/page_30.pdf:3 | 11.2%         | ❌   | ❌         | ✅       |
| Double_EW/2016/page_79.pdf:3   | 921%          | ✅   | ❌         | ✅       |
| Double_GS/2017/page_143.pdf:3  | -26           | ❌   | ✅         | ❌       |

**관찰:** 복잡한 수치 추론에서는 Rewrite와 Selection이 Raw의 오답을 정답으로 바꾸는 사례가 존재했다.

따라서 세 전략의 결과가 완전히 동일하지 않으며, 입력 표현 방식에 따라 문제별 정답 여부가 달라진다.

---

## 7. 평가 한계

### 7.1 EM과 Semantic Correctness의 차이

CoQA의 Accuracy는 Normalized Exact Match로 측정된다.

예를 들어 Gold가 `Miriam`이고 출력이 `It was Miriam.`이면 의미적으로 정답이어도 EM은 0이 된다.

CoQA에서 Raw의 EM은 0.320이지만 Token F1은 0.559이므로, EM만으로 QA 품질을 판단하기 어렵다.

향후에는 다음을 함께 평가하는 것이 적절하다.

- Normalized EM
- Token F1
- Semantic Correctness (LLM-as-Judge 또는 검증된 평가기)

### 7.2 ConvFinQA Numeric Accuracy

ConvFinQA는 상대오차 1% 또는 절대오차 1e-4 이내를 정답으로 인정한다.

이는 엄격한 Numeric Exact Match와 다른 지표이며, 수치 포맷, 단위와 스케일 변환에 따른 오채점 가능성을 점검해야 한다.

### 7.3 Gold History 사용

이전 턴은 모델이 실제 생성한 답변이 아닌 Gold Answer를 사용했다.

따라서 이전 답변 오류가 후속 턴에 전파되는 실제 대화 환경은 평가하지 못했다.

### 7.4 Retrieval 미포함

모든 전략에 동일한 원본 문서를 제공했으므로, 결과는 검색 정확도보다 **질문 및 히스토리 표현 방식의 효과**를 평가한다.

별도의 RAG 환경에서는 History Selection 또는 Rewrite가 Retrieval Recall을 높일 가능성이 있으므로 현재 결과를 직접 일반화하기 어렵다.

---

## 8. 최종 결론

| 평가 항목                      | 가장 유리한 전략                |
| -------------------------- | ------------------------ |
| CoQA EM / Token F1         | **Raw Multi-turn**       |
| ConvFinQA Numeric Accuracy | **Raw Multi-turn**       |
| ConvFinQA Token F1         | One-line Rewrite (근소 차이) |
| 전체 Token 효율                | **Raw Multi-turn**       |
| 전체 Latency                 | **Raw Multi-turn**       |
| Answer Input Token 절감      | One-line Rewrite         |
| 히스토리 턴 수 절감                | One-line Rewrite (0턴)    |

### 핵심 발견

1. **Raw Multi-turn이 두 데이터셋에서 가장 높은 정답률과 가장 낮은 전체 비용을 기록했다.**
2. History Selection은 히스토리 길이를 줄였지만, 정확도가 4\~6%p 낮아졌다.
3. One-line Rewrite는 History Selection보다 높은 정확도를 보였으며, ConvFinQA에서는 Raw와 거의 유사한 성능이었다.
4. 추가 전처리 LLM 호출로 인해 Selection과 Rewrite의 전체 토큰 및 지연시간이 증가했다.
5. 정확도의 통계적 차이는 명확하지 않으므로 더 다양한 데이터와 긴 대화 문맥에서 검증할 필요가 있다.

### 권장 후속 실험

- **Long History:** 이전 턴 수를 5, 10, 20, 50턴으로 늘려 길이에 따른 성능·비용 변화 측정
- **Selective Rewrite:** 현재 질문만으로 해결 가능한 경우 Rewrite 생략
- **Retrieval 포함:** Raw/Selection/Rewrite에 동일한 Dense Retriever를 적용하고 Recall\@k 및 최종 Answer Correctness 평가
- **Semantic Judge:** CoQA에서 표현 차이 때문에 발생하는 EM 오차 분리
- **Failure Analysis:** Selection이 제외한 Q/A와 Rewrite된 질문을 분석하여 실제 정보 손실 여부 확인

**최종 판단:** 이번 100개씩의 실험 조건에서는 Raw Multi-turn을 기본 전략으로 사용하는 것이 합리적이다. History Selection 및 One-line Rㅇewrite는 긴 히스토리나 검색 쿼리 최적화가 중요한 환경에서 추가 검증이 필요하다.
