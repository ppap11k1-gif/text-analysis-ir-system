# 텍스트 데이터 분석 및 검색 시스템 구현 프로젝트

## 핵심 요약

Stack Overflow 태그 데이터를 대상으로 연관 규칙을 분석하고, 학술 논문 검색 엔진의 랭킹 성능을 개선하며, 저널 논문을 자동 분류하는 세 가지 텍스트 분석 파이프라인을 구현했습니다.

---

## 주요 성과

- **검색 엔진 BPref:** 0.2497 → **0.2776** (+11.2% 향상)
- **텍스트 분류 정확도:** SVM **76%**, Naïve Bayes **70%**
- **연관 규칙:** Lift ≥ 2.0 기준 **28개** 규칙 추출

---

## PART I — 연관 규칙 분석 (Association Rule Mining)

Stack Overflow 질문-태그 데이터에서 함께 자주 등장하는 태그 조합을 발굴했습니다.

**설계 결정:**
- 질문 × 태그 수평 행렬(Horizontal Table)로 변환 후 Apriori 적용
- `min_support=0.005`, `min_lift=2.0` — 노이즈를 걸러내고 실질적 연관만 추출
- Lift 임계값을 높일수록 규칙 수는 줄지만 신뢰도는 올라가는 트레이드오프 확인

---

## PART II — 정보 검색 엔진 성능 개선

Whoosh 기반 검색 엔진에서 BPref 점수를 baseline 대비 11.2% 향상시켰습니다.

**개선 전략 (단계별 적용):**

| 개선 항목 | 방법 | 근거 |
|-----------|------|------|
| 인덱싱 | StemmingAnalyzer 적용 | 형태 변화(run/running) 통일로 검색 누락 감소 |
| 질의 전처리 | 불용어(stopword) 제거 | IDF 왜곡 방지 |
| 점수 함수 | BM25 커스텀 구현 (k₁=0.3, b=0.55) | TF 포화를 빠르게 적용해 반복 단어 과가중 방지 |
| 질의 결합 | OrGroup.factory(0.7) | 희귀 단어에 더 높은 가중치 부여 |

**핵심 구현 — BM25 커스텀 스코어링:**
```python
def intappscorer(tf, idf, cf, qf, dc, fl, avgfl, param):
    k1, b = 0.3, 0.55
    norm_tf = (tf * (k1 + 1)) / (tf + k1 * ((1 - b) + b * (fl / avgfl)))
    return idf * norm_tf
```

---

## PART III — 텍스트 자동 분류

4개 학술 저널(`AnnStat`, `Biometrika`, `JASA`, `JMLR`) 논문을 TF-IDF 벡터화 후 분류했습니다.

**모델 비교:**

| 모델 | 정확도 | 주요 설정 |
|------|--------|-----------|
| LinearSVC | **76%** | `C=10.0` |
| MultinomialNB | **70%** | `alpha=0.01`, bigram 추가 |

SVM이 NB보다 높은 성능을 보인 이유: 고차원 TF-IDF 공간에서 클래스 간 결정 경계를 더 정교하게 학습. NB는 단어 독립 가정의 한계를 bigram(연속 2단어 특징) 추가로 일부 보완.

---

## 프로젝트 구조

```
DMA-project2/
├── README.md
├── part1.py           # PART I: 연관 규칙 분석
├── make_index.py      # PART II: 검색 인덱스 생성
├── CustomScoring.py   # PART II: BM25 커스텀 랭킹
├── QueryResult.py     # PART II: 검색 실행
└── clasification.py   # PART III: 텍스트 분류
```

---

## 기술 스택

- **Language:** Python 3.x
- **IR / Search:** Whoosh, NLTK
- **ML / NLP:** scikit-learn (TfidfVectorizer, LinearSVC, MultinomialNB)
- **Association Mining:** mlxtend (Apriori)
- **Data:** pandas

---

## 회고

**강점:** 각 파트에서 단순 구현에 그치지 않고 파라미터 튜닝과 전처리 전략을 직접 설계해 정량적 성능 개선을 달성

**개선 과제:** PART II에서 문서 길이 분포 EDA를 사전에 수행했다면 k₁·b 탐색 범위를 더 효율적으로 좁힐 수 있었을 것
