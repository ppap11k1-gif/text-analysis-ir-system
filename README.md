# 데이터관리와 분석 Project 2 — 팀 13

> **과목**: 데이터관리와 분석  
> **팀**: 13조  
> **구성**: PART I (연관 규칙 분석) · PART II (검색 엔진) · PART III (텍스트 분류)

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [PART I — 연관 규칙 분석 (Association Rule Mining)](#2-part-i--연관-규칙-분석)
3. [PART II — 정보 검색 엔진 (Search Engine)](#3-part-ii--정보-검색-엔진)
4. [PART III — 텍스트 분류 (Text Classification)](#4-part-iii--텍스트-분류)
5. [실행 방법](#5-실행-방법)
6. [결과 요약](#6-결과-요약)

---

## 1. 프로젝트 개요

이 프로젝트는 세 가지 독립적인 데이터 관리·분석 과제로 구성됩니다.

| 파트 | 주제 | 핵심 기법 |
|------|------|-----------|
| PART I | Stack Overflow 태그 연관 규칙 분석 | Apriori, Support/Confidence/Lift |
| PART II | 학술 논문 검색 엔진 성능 개선 | BM25, TF-IDF, Stemming, BPref |
| PART III | 학술 저널 텍스트 자동 분류 | Naïve Bayes, SVM, TF-IDF |

---

## 2. PART I — 연관 규칙 분석

### 2.1 문제 정의

Stack Overflow의 질문-태그 데이터(`DMA_project_UBR.csv`)에서 **자주 함께 사용되는 태그 조합**을 발굴하고, 의미 있는 연관 규칙을 추출합니다.

### 2.2 방법론

**연관 규칙 분석(Association Rule Mining)**은 장바구니 분석에서 유래한 기법으로, "A를 구매한 사람이 B도 함께 구매한다"는 패턴을 찾습니다. 여기서는 "특정 태그를 단 질문이 다른 특정 태그도 함께 갖는" 패턴을 분석합니다.

#### 핵심 지표

| 지표 | 수식 | 의미 |
|------|------|------|
| **Support** | P(A ∩ B) | 전체 질문 중 A와 B가 함께 등장하는 비율 |
| **Confidence** | P(B\|A) = P(A ∩ B) / P(A) | A가 있을 때 B도 있을 조건부 확률 |
| **Lift** | Confidence / P(B) | 독립 대비 실제 연관 강도 (1 초과 시 양의 상관) |

- **Lift > 1**: 두 태그가 우연보다 더 자주 함께 등장 → 의미 있는 연관
- **Lift = 1**: 통계적으로 독립 (연관 없음)
- **Lift < 1**: 음의 상관 (함께 잘 안 나옴)

#### 처리 흐름

```
CSV 로드
  ↓
수평 테이블(Horizontal Table) 변환
  → 행: 질문(question ID), 열: 태그(tag), 값: True/False
  ↓
Apriori 알고리즘으로 빈발 항목 집합 추출
  → min_support = 0.005 (전체 질문의 0.5% 이상 등장)
  ↓
연관 규칙 생성
  → metric = 'lift', min_threshold = 2.0
  ↓
결과 저장 (pkl)
```

### 2.3 구현

```python
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv('DMA_project_UBR.csv')
df['value'] = True

# 수평 테이블 변환: 질문 × 태그 행렬
horizontal = df.pivot_table(
    index='question', columns='tag', values='value', aggfunc='any'
).fillna(False)

# 빈발 항목 집합 추출
frequent_itemsets = apriori(horizontal, min_support=0.005, use_colnames=True)

# 연관 규칙 생성 (lift ≥ 2.0)
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=2.0)
```

### 2.4 결과

- 추출된 연관 규칙 수: **28개** (lift ≥ 2.0 기준)
- 임계값(min_threshold) 변화에 따른 규칙 수 변화:

| Lift 임계값 | 추출 규칙 수 | 비고 |
|-------------|-------------|------|
| 1.0 | 매우 많음 | 통계적 독립 포함, 노이즈 많음 |
| 2.0 | **28개** | 선택값: 실질적 연관만 추출 |
| 3.0 | 적음 | 매우 강한 연관만 남음 |

- 임계값이 높을수록 규칙의 수는 줄지만 **품질(신뢰도)은 높아집니다**.
- 이 분석 결과는 누가 실행해도 동일 데이터·동일 파라미터라면 **동일한 결과**가 나옵니다 (결정론적 알고리즘).

---

## 3. PART II — 정보 검색 엔진

### 3.1 문제 정의

학술 논문 컬렉션에서 질의(query)를 입력하면 관련 논문을 순위별로 반환하는 검색 엔진을 구현합니다. 평가 지표는 **BPref(Binary Preference)**를 사용합니다.

### 3.2 BPref 평가 지표

$$\text{BPref} = \frac{1}{R} \sum_{r \in R} \left(1 - \frac{|\text{non-relevant before } r|}{R}\right)$$

- R: 관련 문서의 수
- 관련 문서가 비관련 문서보다 **앞에 랭킹**될수록 높은 점수
- 0~1 사이 값, 높을수록 좋은 성능

### 3.3 개선 전략과 이론적 근거

기본 제공 모델(BM25F, BPref = 0.2497)을 출발점으로 세 가지 측면에서 개선했습니다.

#### ① 인덱스 개선 — StemmingAnalyzer 적용

**이론**: 같은 의미의 단어도 형태가 다르면 (run / running / ran) 서로 다른 토큰으로 처리되어 검색 누락이 발생합니다. **Porter Stemmer**는 단어를 어간(stem)으로 변환하여 형태 변화를 통일합니다.

```
running → run
studies → studi
statistical → statist
```

`make_index.py`에서 `StemmingAnalyzer`를 스키마에 적용하면, 색인 시점에 자동으로 어간 변환이 적용됩니다. QueryParser도 같은 스키마를 사용하므로 질의어도 자동으로 동일하게 변환됩니다.

```python
from whoosh.analysis import StemmingAnalyzer
schema = Schema(
    docID=NUMERIC(stored=True),
    contents=TEXT(analyzer=StemmingAnalyzer())
)
```

#### ② 질의어 전처리 — 불용어(Stopword) 제거

**이론**: "the", "is", "a", "of" 같은 불용어는 검색 성능에 기여하지 않으면서 IDF 가중치를 왜곡합니다. NLTK의 영어 불용어 목록으로 제거합니다.

```python
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))
new_q = ' '.join(w for w in query.lower().split() if w not in stopWords)
```

#### ③ 점수 함수 개선 — BM25 튜닝

**이론**: BM25(Okapi BM25)는 TF-IDF를 확률적으로 개선한 랭킹 모델입니다.

$$\text{BM25}(q, d) = \text{IDF}(q) \cdot \frac{\text{TF}(q, d) \cdot (k_1 + 1)}{\text{TF}(q, d) + k_1 \cdot \left((1-b) + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

| 파라미터 | 역할 | 튜닝 결과 |
|---------|------|-----------|
| k₁ | TF 포화 파라미터. 높을수록 TF 반복의 영향이 커짐 | **0.3** (낮은 값 → 빠른 포화, 짧은 문서에 유리) |
| b | 문서 길이 정규화. 1이면 완전 정규화, 0이면 없음 | **0.55** |

기본 제공 BM25F의 기본값(k₁=1.2, b=0.75)보다 k₁을 낮춘 이유: 학술 논문은 같은 단어가 반복되어도 단순 반복이 아닌 중요도를 의미하는 경우가 많아, 포화를 더 빠르게 적용하는 것이 효과적이었습니다.

#### ④ 질의 결합 방식 — OrGroup.factory(0.7)

Whoosh의 `OrGroup`은 여러 단어를 OR 조건으로 검색합니다. `OrGroup.factory(0.7)`은 각 단어의 IDF 가중치에 0.7의 스케일 팩터를 적용하여, **희귀한 단어에 더 높은 가중치**를 부여합니다. 이를 통해 흔한 단어가 점수를 과도하게 끌어올리는 것을 방지합니다.

```python
parser = QueryParser("contents", schema=ix.schema, group=OrGroup.factory(0.7))
```

### 3.4 성능 비교

| 설정 | BPref 점수 |
|------|-----------|
| 기본 BM25F (baseline) | 0.2497 |
| + StemmingAnalyzer | 0.2600 수준 |
| + 불용어 제거 | 0.2700 수준 |
| + BM25 커스텀 (k₁=0.3, b=0.55) | 0.2750 수준 |
| **+ OrGroup.factory(0.7) (최종)** | **0.2776** |

**baseline 대비 +11.2% 향상** (0.2497 → 0.2776)

### 3.5 최종 파라미터

```python
# CustomScoring.py
k1 = 0.3
b  = 0.55

# QueryResult.py
parser = QueryParser("contents", schema=ix.schema, group=OrGroup.factory(0.7))
```

---

## 4. PART III — 텍스트 분류

### 4.1 문제 정의

4개의 학술 저널 논문을 학술지 카테고리(`AnnStat`, `Biometrika`, `JASA`, `JMLR`)로 자동 분류합니다.

### 4.2 방법론

#### TF-IDF 벡터화

텍스트를 수치 벡터로 변환하는 핵심 전처리 단계입니다.

- **TF(Term Frequency)**: 해당 문서 안에서 단어의 등장 빈도
- **IDF(Inverse Document Frequency)**: 단어의 희귀성 (드물수록 높음)
- **TF-IDF**: 자주 등장하지만 여러 문서에 퍼져 있지 않은 단어를 높게 평가

`sublinear_tf=True`: log(TF + 1)을 적용하여 빈도 급증을 완화합니다.

#### 분류 모델

**모델 1 — Naïve Bayes (MultinomialNB)**

베이즈 정리를 기반으로 한 확률 모델입니다. 각 단어가 독립이라는 가정 하에 문서가 각 클래스에 속할 사후 확률을 계산합니다.

$$P(c|d) \propto P(c) \prod_{i} P(t_i|c)$$

- `alpha=0.01`: 라플라스 스무딩 파라미터. 훈련에 없던 단어에 0이 아닌 작은 확률 부여
- `ngram_range=(1,2)`: 단어 단독(unigram)뿐 아니라 인접한 두 단어 조합(bigram)도 특징으로 사용 → 문맥 파악 능력 향상

**모델 2 — SVM (LinearSVC)**

고차원 특징 공간에서 클래스 간 결정 경계(hyperplane)를 최대 마진으로 학습합니다. 텍스트 분류에서 높은 성능을 보이는 대표적 모델입니다.

$$\min_{w,b} \frac{1}{2}||w||^2 + C \sum_i \max(0, 1 - y_i(w^T x_i + b))$$

- `C=10.0`: 규제 파라미터. 훈련 데이터에 더 엄격하게 맞추도록 설정

### 4.3 파이프라인 구조

```
텍스트 입력
    ↓
TfidfVectorizer
  - sublinear_tf=True
  - min_df=2 (2개 이상 문서 등장 단어만)
  - stop_words='english'
  - [NB만] ngram_range=(1,2)
    ↓
분류기 (MultinomialNB / LinearSVC)
    ↓
클래스 예측 (AnnStat / Biometrika / JASA / JMLR)
```

### 4.4 결과

| 모델 | 정확도 |
|------|--------|
| Naïve Bayes (baseline) | 65% |
| **Naïve Bayes (개선: bigram)** | **70%** |
| SVM (baseline) | 75% |
| **SVM (개선: C=10.0)** | **76%** |

**SVM이 NB보다 높은 성능**을 보입니다. 이는 SVM이 고차원 TF-IDF 공간에서 결정 경계를 더 정교하게 학습할 수 있기 때문입니다. NB는 단어 독립 가정이 실제와 다를 때 성능이 제한되지만, bigram 추가로 문맥 정보를 보완했습니다.

#### 혼동 행렬 해석 (NB)

혼동 행렬(Confusion Matrix)의 대각선 원소가 클수록 올바른 분류, 비대각선 원소는 오분류를 나타냅니다. `JMLR`(머신러닝 저널)은 다른 통계학 저널(`AnnStat`, `Biometrika`, `JASA`)과 단어 분포가 달라 상대적으로 구분이 용이합니다.

---

## 5. 실행 방법

### PART I

```bash
cd AA
python part1.py
# 출력: DMA_project2_team13_part1_horizontal.pkl
#       DMA_project2_team13_part1_association.pkl
```

### PART II

```bash
cd SE
# 1. 인덱스 생성 (최초 1회)
python make_index.py

# 2. 평가 실행
python evaluate.py
# 출력: BPref 점수 (0.2776)
```

### PART III

```bash
cd CL
python clasification.py
# 출력: DMA_project2_team13_nb.pkl
#       DMA_project2_team13_svm.pkl
```

### 의존성 설치

```bash
pip install scikit-learn mlxtend whoosh nltk
python -c "import nltk; nltk.download('stopwords')"
```

---

## 6. 결과 요약

| 파트 | 지표 | 결과 |
|------|------|------|
| PART I | 추출 연관 규칙 수 (lift ≥ 2.0) | **28개** |
| PART II | BPref 점수 | **0.2776** (baseline 0.2497 대비 +11.2%) |
| PART III | NB 정확도 | **70%** |
| PART III | SVM 정확도 | **76%** |

---

## 기술 스택

- **Python** 3.x
- **scikit-learn** — TF-IDF, Naïve Bayes, SVM, Pipeline
- **mlxtend** — Apriori, Association Rules
- **Whoosh** — 풀텍스트 검색 인덱싱 및 랭킹
- **NLTK** — 불용어 목록
- **pandas** — 데이터 전처리 및 피벗 테이블
