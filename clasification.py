from sklearn.datasets import load_files
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn import metrics
import numpy as np
import pickle

# ============================================================
# !! TODO - 팀 번호를 수정하세요 !!
# ============================================================
TEAM = 13

# 데이터 설정
categories = ['AnnStat', 'Biometrika', 'JASA', 'JMLR']

train_data = load_files(container_path='text/train', categories=categories,
                        shuffle=True, encoding='utf-8', decode_error='replace')

# ============================================================
# 3-1. Naive Bayes Classifier Pipeline
# - TfidfVectorizer: 텍스트를 TF-IDF 수치 벡터로 변환
#   - sublinear_tf: log 스케일 TF 사용 (빈도 급증 완화)
#   - min_df=2: 2개 이상 문서에 등장한 단어만 사용
#   - stop_words: 영어 불용어 제거
# - MultinomialNB: 단어 빈도 기반 나이브 베이즈 분류기
#   - alpha: 스무딩 파라미터 (0에 가까울수록 날카로운 분류)
# ============================================================
clf_nb = Pipeline([
    ('tfidf', TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words='english', ngram_range=(1,2))),
    ('clf', MultinomialNB(alpha=0.01))
])
clf_nb.fit(train_data.data, train_data.target)

# ============================================================
# 3-2. SVM Classifier Pipeline
# - TfidfVectorizer: 동일한 텍스트 전처리
# - LinearSVC: 선형 SVM 분류기
#   - C: 규제 파라미터 (높을수록 훈련 데이터에 더 맞춤)
# ============================================================
clf_svm = Pipeline([
    ('tfidf', TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words='english')),
    ('clf', LinearSVC(C=10.0))
])
clf_svm.fit(train_data.data, train_data.target)

# ╔════════════════════════════════════════════════════════════╗
# ║  DO NOT MODIFY BELOW - 평가 및 모델 저장                  ║
# ╚════════════════════════════════════════════════════════════╝
test_data = load_files(container_path='text/test', categories=categories,
                       shuffle=True, encoding='utf-8', decode_error='replace')
docs_test = test_data.data

# Evaluate Naive Bayes
predicted = clf_nb.predict(docs_test)
print("NB accuracy : %d / %d" % (np.sum(predicted==test_data.target), len(test_data.target)))
print(metrics.classification_report(test_data.target, predicted, target_names=test_data.target_names))
print(metrics.confusion_matrix(test_data.target, predicted))

# Evaluate SVM
predicted = clf_svm.predict(docs_test)
print("\nSVM accuracy : %d / %d" % (np.sum(predicted==test_data.target), len(test_data.target)))
print(metrics.classification_report(test_data.target, predicted, target_names=test_data.target_names))
print(metrics.confusion_matrix(test_data.target, predicted))

# Save models
with open('DMA_project2_team%02d_nb.pkl' % TEAM, 'wb') as f1:
    pickle.dump(clf_nb, f1)

with open('DMA_project2_team%02d_svm.pkl' % TEAM, 'wb') as f2:
    pickle.dump(clf_svm, f2)
