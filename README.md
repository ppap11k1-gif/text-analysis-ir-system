# 텍스트 데이터 분석 및 검색 시스템

> Stack Overflow 태그 연관 분석 · 학술 논문 검색 엔진 개선 · 저널 자동 분류

---

## 주요 성과

| | 지표 | 결과 |
|---|---|---|
| 🔍 검색 엔진 | BPref | 0.2497 → **0.2776** (+11.2%) |
| 🤖 텍스트 분류 | SVM 정확도 | **76%** |
| 🤖 텍스트 분류 | NB 정확도 | **70%** |
| 🔗 연관 규칙 | 추출 규칙 수 | **28개** (Lift ≥ 2.0) |

---

## PART I — 연관 규칙 분석

Stack Overflow 질문-태그 데이터에서 함께 자주 등장하는 태그 조합을 발굴했습니다.

- 질문 × 태그 수평 행렬로 변환 후 Apriori 알고리즘 적용
- `min_support=0.005` / `min_lift=2.0` 으로 노이즈 제거, 실질적 연관만 추출
- Lift가 높을수록 규칙 수는 줄지만 신뢰도는 상승하는 트레이드오프 확인

---

## PART II — 검색 엔진 랭킹 개선

Whoosh 기반 검색 엔진의 BPref를 baseline 대비 **11.2% 향상**시켰습니다.

| 개선 항목 | 방법 | 효과 |
|---|---|---|
| 인덱싱 | StemmingAnalyzer | run/running 등 형태 통일 → 검색 누락 감소 |
| 질의 전처리 | 불용어 제거 | IDF 왜곡 방지 |
| 점수 함수 | BM25 커스텀 (k₁=0.3, b=0.55) | 반복 단어 과가중 방지 |
| 질의 결합 | OrGroup.factory(0.7) | 희귀 단어 가중치 상향 |

```python
# BM25 커스텀 스코어링
def intappscorer(tf, idf, cf, qf, dc, fl, avgfl, param):
    k1, b = 0.3, 0.55
    norm_tf = (tf * (k1 + 1)) / (tf + k1 * ((1 - b) + b * (fl / avgfl)))
    return idf * norm_tf
```

---

## PART III — 텍스트 자동 분류

4개 학술 저널 논문(`AnnStat` / `Biometrika` / `JASA` / `JMLR`)을 TF-IDF 벡터화 후 분류했습니다.

| 모델 | 정확도 | 주요 설정 |
|---|---|---|
| LinearSVC | **76%** | `C=10.0` |
| MultinomialNB | **70%** | `alpha=0.01` + bigram |

SVM은 고차원 TF-IDF 공간에서 결정 경계를 정교하게 학습해 NB 대비 우수한 성능을 보였습니다.
NB는 bigram 추가로 단어 독립 가정의 한계를 일부 보완했습니다.

---

## 프로젝트 구조

```
├── part1.py           # PART I  : 연관 규칙 분석
├── make_index.py      # PART II : 검색 인덱스 생성
├── CustomScoring.py   # PART II : BM25 커스텀 랭킹
├── QueryResult.py     # PART II : 검색 실행
└── clasification.py   # PART III: 텍스트 분류
```

---

## 기술 스택

`Python` `scikit-learn` `Whoosh` `NLTK` `mlxtend` `pandas`

---

## 회고

**잘한 점** 단순 구현을 넘어 파라미터 튜닝과 전처리 전략을 직접 설계해 정량적 성능 개선을 달성했습니다.

**아쉬운 점** 사전 EDA로 문서 길이 분포를 파악했다면 k₁·b 탐색 범위를 더 효율적으로 좁힐 수 있었을 것입니다.
