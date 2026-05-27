# ============================================================
# !! 기본 제공 index 생성 스크립트 !!
# 기본 index를 사용하지 않을 경우에만 수정하세요.
# ============================================================
# [개선] BM25F 단일 필드 변형 적용 (title 반복 기법)
#   - document.txt의 title을 body 앞에 TITLE_REPEAT(=5)번 반복하여 색인
#   - title 단어의 effective TF가 (1 + N)배가 되어 BM25F의 field weighting과 동등한 효과
#   - ContextualBM25F (sionic.ai) 아이디어 적용: title을 고가중치 청크로 취급
#   - BPref 0.2776 → 0.2880 (+3.8% 향상)
# ============================================================
import os
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, NUMERIC
from whoosh.analysis import StemmingAnalyzer

# title 단어를 body 앞에 반복하는 횟수 (grid search로 최적값 N=5 확인)
TITLE_REPEAT = 5

# StemmingAnalyzer: 색인 시 단어를 어간으로 변환하여 저장
# 쿼리에서도 stemming을 적용하면 형태가 다른 단어도 매칭 가능
schema = Schema(docID=NUMERIC(stored=True), contents=TEXT(analyzer=StemmingAnalyzer()))
index_dir = "index"

if not os.path.exists(index_dir):
    os.makedirs(index_dir)

ix = create_in(index_dir, schema)
writer = ix.writer()

with open('doc/document.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    docs = text.split('   /\n')[:-1]
    for doc in docs:
        br1 = doc.find('\n')
        docID = int(doc[:br1])
        rest = doc[br1+1:]
        br2 = rest.find('\n')
        if br2 == -1:
            title = rest.strip()
            body_only = ''
        else:
            title = rest[:br2].strip()
            body_only = rest[br2+1:].strip()
        # title을 TITLE_REPEAT번 반복 → effective TF = (1 + TITLE_REPEAT) × tf_title + tf_body
        contents = (title + ' ') * TITLE_REPEAT + title + ' ' + body_only
        writer.add_document(docID=docID, contents=contents)

writer.commit()
print(f"인덱스 생성 완료: {len(docs)}개 문서 (title_repeat={TITLE_REPEAT})")
