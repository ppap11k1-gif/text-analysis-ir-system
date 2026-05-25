# ============================================================
# !! 기본 제공 index 생성 스크립트 !!
# 기본 index를 사용하지 않을 경우에만 수정하세요.
# ============================================================
import os.path
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, NUMERIC
from whoosh.analysis import StemmingAnalyzer

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
        br = doc.find('\n')
        docID = int(doc[:br])
        doc_text = doc[br+1:]
        writer.add_document(docID=docID, contents=doc_text)

writer.commit()
print(f"인덱스 생성 완료: {len(docs)}개 문서")