# 📊 텍스트 데이터 분석 및 검색 시스템

> Stack Overflow 태그 연관 분석 · 학술 논문 검색 엔진 개선 · 저널 자동 분류

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Whoosh](https://img.shields.io/badge/Whoosh-IR-4CAF50?style=flat)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📊 핵심 성과

| 검색 엔진 BPref | SVM 정확도 | NB 정확도 | 연관 규칙 수 |
|:---:|:---:|:---:|:---:|
| 0.2497 → **0.2880** (**+15.34%**) | **76%** | **70%** | **28개** (Lift ≥ 2.0) |

---

## 📋 목차

- [프로젝트 개요](#-프로젝트-개요)
- [PART I — 연관 규칙 분석](#part-i--연관-규칙-분석)
- [PART II — 검색 엔진 랭킹 개선](#part-ii--검색-엔진-랭킹-개선)
- [PART III — 텍스트 자동 분류](#part-iii--텍스트-자동-분류)
- [실행 방법](#-실행-방법)
- [프로젝트 구조](#-프로젝트-구조)
- [기술 스택](#️-기술-스택)
- [회고](#-회고)

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **과목** | 데이터관리와 분석 (2026-1학기) |
| **과제** | Project 2 — 텍스트 데이터 분석 |
| **기간** | 2026.04 ~ 2026.05 |

Stack Overflow 태그 데이터에서 연관 규칙을 발굴하고, 학술 논문 검색 엔진의 랭킹을 개선하며, TF-IDF 기반 저널 자동 분류 모델을 구현한 프로젝트입니다.

---

## PART I — 연관 규칙 분석

Stack Overflow 질문-태그 데이터에서 함께 자주 등장하는 태그 조합을 발굴했습니다.

### 분석 방법

- 질문 × 태그 수평 행렬로 변환 후 **Apriori 알고리즘** 적용
- `min_support=0.005` / `min_lift=2.0` 으로 노이즈 제거, 실질적 연관만 추출
- Lift가 높을수록 규칙 수는 줄지만 신뢰도는 상승하는 트레이드오프 확인

### 주요 발견

- Lift ≥ 2.0 기준으로 **28개** 유의미한 연관 규칙 추출
- `anova` ↔ `repeated-measures` (lift=11.06), `arima` ↔ `time-series` (lift=9.49) 등 강한 공출현 패턴 확인
- Lift 임계값별 규칙 수 변화 정량 분석 (0.5~10.0 구간)

---

## PART II — 검색 엔진 랭킹 개선

Whoosh 기반 검색 엔진의 BPref를 baseline 대비 **+15.34% 향상**시켰습니다.  
**과제 요구사항을 넘어 능동적으로 BM25F 개념을 적용**한 파트입니다.

### 개선 전략 전체 흐름

```
Baseline (BM25F default)          BPref 0.2497
    ↓ 1차 개선 (일반 IR 튜닝)
StemmingAnalyzer + BM25 파라미터  BPref 0.2776  (+11.2%)
    ↓ 2차 개선 (BM25F 자체 적용)
단일 필드 BM25F 등가 구현          BPref 0.2880  (+15.34%)
```

### 1차 개선 — 일반 IR 튜닝 (+11.2%)

| 단계 | 구성 | BPref | Δ |
|---|---|---|---|
| 0 | BM25F default | 0.2497 | — |
| 1 | + Porter StemmingAnalyzer | 0.2648 | +0.0151 |
| 2 | + 질의어 불용어 제거 | 0.2648 | +0.0000 |
| 3 | + BM25 커스텀 (k₁=0.3, b=0.55) | 0.2773 | +0.0125 |
| 4 | + OrGroup.factory(0.7) | 0.2776 | +0.0003 |

### 2차 개선 — BM25F 자체 적용 시도와 우회 (+3.75%) ⭐

> **이 부분은 과제 요구사항에 없었으나, 성능 한계를 돌파하기 위해 능동적으로 탐색한 과정입니다.**

**막힌 지점:**  
표준적인 BM25F(멀티필드 분리)를 적용하자 BPref가 오히려 0.2776 → 0.2391로 하락했습니다.  
원인: `MultifieldParser`가 단어 2개짜리 쿼리를 필드 조합 4개짜리로 분해하면서, `OrGroup(0.7)`의 감쇠 구조가 의도와 다르게 작동한 것.

```
단일필드: score(bayesian) + score(regression) × 0.7  ← 의도대로
멀티필드: score(title:bayesian) + score(body:bayesian) × 0.7
        + score(title:regression) × 0.7² + score(body:regression) × 0.7³  ← 뒤죽박죽
```

**전환점 — sionic.ai 테크블로그 발견:**  
[ContextualBM25F 기술 블로그](https://blog.sionic.ai/introducing-contextual-bm25f)에서 핵심 통찰을 얻었습니다.

> "BM25F의 본질은 필드를 물리적으로 분리하는 것이 아니라,  
> 특정 신호의 effective TF를 가중하는 것이다."

**해결책 — 단일 필드 내 BM25F 등가 구현:**

```python
# make_index.py 핵심 로직
TITLE_REPEAT = 5
contents = (title + ' ') * TITLE_REPEAT + title + ' ' + body_only
```

제목 단어를 본문 앞에 N번 반복 색인 → effective TF = (1+N)×tf_title + tf_body  
→ 표준 BM25F의 field weighting과 수식적으로 동일하면서, OrGroup 구조는 그대로 유지.

**Grid Search 결과:**

| title_repeat | BPref |
|:---:|:---:|
| 0 (baseline) | 0.2776 |
| 1 | 0.2842 |
| 3 | 0.2874 |
| **5 ← 최적** | **0.2880** |
| 8 | 0.2873 |
| 20 | 0.2866 |

**추가 실험 — 청크 거리 가중치 (음의 결과도 기록):**  
sionic.ai 블로그의 두 번째 아이디어(청크 거리 가중치)도 14가지 조합으로 실험했습니다.  
결과: 최대 +0.0007 (노이즈 수준) → 학술 문서 단위 검색에서는 효과 없음을 정량 확인.  
음의 결과도 의사결정의 근거로 문서화했습니다.

### 핵심 구현 — BM25 커스텀 스코어링

```python
def intappscorer(tf, idf, cf, qf, dc, fl, avgfl, param):
    k1, b = 0.3, 0.55
    norm_tf = (tf * (k1 + 1)) / (tf + k1 * ((1 - b) + b * (fl / avgfl)))
    return idf * norm_tf
```

---

## PART III — 텍스트 자동 분류

4개 학술 저널 논문(`AnnStat` / `Biometrika` / `JASA` / `JMLR`)을 TF-IDF 벡터화 후 분류했습니다.

### 모델 비교

| 모델 | 정확도 | 주요 설정 |
|---|---|---|
| **LinearSVC** | **76%** | `C=10.0` |
| MultinomialNB | 70% | `alpha=0.01`, `ngram_range=(1,2)` |

- **SVM**: 고차원 TF-IDF 공간에서 결정 경계를 정교하게 학습 → NB 대비 우수한 성능
- **NB**: bigram 추가(`ngram_range=(1,2)`)로 단어 독립 가정의 한계를 일부 보완

---

## 🚀 실행 방법

```bash
pip install scikit-learn whoosh nltk mlxtend pandas
```

```bash
# PART I — 연관 규칙 분석
cd 작업/AA
python part1.py

# PART II — 검색 인덱스 생성 + 평가
cd 작업/SE
python make_index.py
python evaluate.py

# PART III — 텍스트 분류
cd 작업/CL
python clasification.py
```

---

## 📁 프로젝트 구조

```
project 2/
├── 작업/
│   ├── AA/
│   │   └── part1.py              # PART I  : 연관 규칙 분석 (Apriori)
│   ├── SE/
│   │   ├── make_index.py         # PART II : 검색 인덱스 생성 (title 반복 BM25F)
│   │   ├── CustomScoring.py      # PART II : BM25 커스텀 랭킹 함수
│   │   ├── QueryResult.py        # PART II : 검색 실행
│   │   ├── evaluate.py           # PART II : BPref 평가
│   │   ├── BM25F_실험/           # title 반복 기법 실험 파일들
│   │   └── chunk_실험/           # 청크 거리 가중치 실험 파일들
│   └── CL/
│       └── clasification.py      # PART III: TF-IDF 기반 텍스트 분류
└── sionic_internship_appeal.txt  # 개선 과정 상세 기록
```

---

## 🛠️ 기술 스택

`Python` `scikit-learn` `Whoosh` `NLTK` `mlxtend` `pandas`

---

## 💬 회고

**잘한 점**  
과제 요구사항인 BM25 구현에 머물지 않고, 성능 한계를 만났을 때 외부 자료(sionic.ai 테크블로그)를 능동적으로 탐색했습니다. 표준 BM25F 적용이 실패했을 때 원인을 분석하고, 라이브러리 제약 안에서 동등한 효과를 내는 우회 방법을 직접 설계해 정량적으로 검증했습니다. 음의 실험 결과(청크 거리 가중치)도 가설 검증의 일부로 문서화한 점도 의미 있었습니다.

**아쉬운 점**  
사전 EDA로 문서 길이 분포를 파악했다면 k₁·b 탐색 범위를 더 효율적으로 좁힐 수 있었을 것입니다. 분류 모델에서도 교차검증 기반 하이퍼파라미터 탐색을 적용했다면 더 신뢰할 수 있는 성능 비교가 가능했을 것입니다.
