import whoosh.index as index
from whoosh.qparser import QueryParser, OrGroup
from whoosh import scoring
import CustomScoring as scoring
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def getSearchEngineResult(query_dict):
    """
    질의어 딕셔너리를 받아 검색 결과를 반환한다.
    - query_dict: {queryID: query_text, ...}
    - 반환값: {queryID: [docID, docID, ...], ...}  (관련도 순)
    """
    result_dict = {}
    ix = index.open_dir("index")

    # ============================================================
    # Custom scoring 사용 (BM25 기반 커스텀 스코어링)
    # ============================================================
    # with ix.searcher(weighting=scoring.BM25F()) as searcher:
    with ix.searcher(weighting=scoring.ScoringFunction()) as searcher:

        # ============================================================
        # 질의어 전처리: stopword 제거 + Porter Stemming 적용
        # - stopword 제거: 의미 없는 단어(the, is, a 등) 제거
        # - stemming: 단어를 어간으로 변환 (running→run, studies→studi)
        #             문서와 쿼리의 단어 형태를 통일시켜 검색 성능 향상
        # ============================================================
        # StemmingAnalyzer가 적용된 schema이므로 QueryParser가 자동으로 stemming 처리
        parser = QueryParser("contents", schema=ix.schema, group=OrGroup.factory(0.7))
        stopWords = set(stopwords.words('english'))

        for qid, q in query_dict.items():
            new_q = ''
            for word in q.split(' '):
                word = word.lower().strip()
                if word and word not in stopWords:
                    new_q += word + ' '
            query = parser.parse(new_q.strip())

            # !! DO NOT MODIFY - 검색 실행 및 결과 수집 !!
            results = searcher.search(query, limit=None)
            result_dict[qid] = [result.fields()['docID'] for result in results]

    return result_dict
