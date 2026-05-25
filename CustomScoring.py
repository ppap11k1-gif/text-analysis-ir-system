# Copyright 2008 Matt Chaput. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY MATT CHAPUT ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL MATT CHAPUT OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are
# those of the authors and should not be interpreted as representing official
# policies, either expressed or implied, of Matt Chaput.

"""
This module contains classes for scoring (and sorting) search results.

[구조 설명]
- intappscorer(): 학생이 작성하는 scoring function (!! TODO !!)
- ScoringFunction / CustomScoring: intappscorer()를 호출하는 프레임워크 (수정 금지)
- BM25F: 기본 제공 baseline 모델 (참고용, 수정 금지)
"""

from __future__ import division
from math import log, pi

from whoosh.compat import iteritems


# ╔════════════════════════════════════════════════════════════╗
# ║  DO NOT MODIFY - whoosh scoring 기반 클래스               ║
# ╚════════════════════════════════════════════════════════════╝

class WeightingModel(object):
    use_final = False

    def idf(self, searcher, fieldname, text):
        parent = searcher.get_parent()
        n = parent.doc_frequency(fieldname, text)
        dc = parent.doc_count_all()
        return log(dc / (n + 1)) + 1

    def scorer(self, searcher, fieldname, text, qf=1):
        raise NotImplementedError(self.__class__.__name__)

    def final(self, searcher, docnum, score):
        return score


class BaseScorer(object):
    def supports_block_quality(self):
        return False

    def score(self, matcher):
        raise NotImplementedError(self.__class__.__name__)

    def max_quality(self):
        raise NotImplementedError(self.__class__.__name__)

    def block_quality(self, matcher):
        raise NotImplementedError(self.__class__.__name__)


class WeightScorer(BaseScorer):
    def __init__(self, maxweight):
        self._maxweight = maxweight

    def supports_block_quality(self):
        return True

    def score(self, matcher):
        return matcher.weight()

    def max_quality(self):
        return self._maxweight

    def block_quality(self, matcher):
        return matcher.block_max_weight()

    @classmethod
    def for_(cls, searcher, fieldname, text):
        ti = searcher.term_info(fieldname, text)
        return cls(ti.max_weight())


class WeightLengthScorer(BaseScorer):
    def setup(self, searcher, fieldname, text):
        ti = searcher.term_info(fieldname, text)
        if not searcher.schema[fieldname].scorable:
            return WeightScorer(ti.max_weight())

        self.dfl = lambda docid: searcher.doc_field_length(docid, fieldname, 1)
        self._maxquality = self._score(ti.max_weight(), ti.min_length())

    def supports_block_quality(self):
        return True

    def score(self, matcher):
        return self._score(matcher.weight(), self.dfl(matcher.id()))

    def max_quality(self):
        return self._maxquality

    def block_quality(self, matcher):
        return self._score(matcher.block_max_weight(), matcher.block_min_length())

    def _score(self, weight, length):
        raise NotImplementedError(self.__wclass__.__name__)


# ╔════════════════════════════════════════════════════════════╗
# ║  !! TODO - 이 함수만 수정하세요 !!                        ║
# ║  자신만의 scoring function을 구현하세요.                   ║
# ║  함수 이름과 파라미터는 변경하지 마세요.                    ║
# ╚════════════════════════════════════════════════════════════╝

def intappscorer(tf, idf, cf, qf, dc, fl, avgfl, param):
    # tf    - term frequency in the current document  (문서 내 단어 빈도)
    # idf   - inverse document frequency              (역문서 빈도)
    # cf    - term frequency in the collection         (전체 컬렉션 내 단어 빈도)
    # qf    - term frequency in the query              (질의어 내 단어 빈도)
    # dc    - doc count                                (전체 문서 수)
    # fl    - field length in the current document     (현재 문서 길이)
    # avgfl - average field length across documents    (평균 문서 길이)
    # param - free parameter                           (자유 파라미터)

    # BM25 기반 커스텀 스코어링
    # k1: term frequency 포화 파라미터 (높을수록 tf 영향 커짐)
    # b:  문서 길이 정규화 파라미터 (1이면 완전 정규화, 0이면 정규화 없음)
    k1 = 0.3
    b = 0.55

    # 문서 길이 정규화된 tf 계산
    norm_tf = (tf * (k1 + 1)) / (tf + k1 * ((1 - b) + b * (fl / avgfl)))

    return idf * norm_tf


# ╔════════════════════════════════════════════════════════════╗
# ║  DO NOT MODIFY - intappscorer()를 호출하는 프레임워크      ║
# ║  QueryResult.py에서 scoring.ScoringFunction()으로 사용     ║
# ╚════════════════════════════════════════════════════════════╝

class ScoringFunction(WeightingModel):
    def __init__(self, param=1.0):
        self.param = param

    def scorer(self, searcher, fieldname, text, qf=1):
        if not searcher.schema[fieldname].scorable:
            return WeightScorer.for_(searcher, fieldname, text)

        return CustomScoring(searcher, fieldname, text, self.param, qf=qf)


class CustomScoring(WeightLengthScorer):
    def __init__(self, searcher, fieldname, text, param, qf=1):
        parent = searcher.get_parent()
        self.idf = parent.idf(fieldname, text)
        self.cf = parent.frequency(fieldname, text)
        self.dc = parent.doc_count_all()
        self.avgfl = parent.avg_field_length(fieldname) or 1

        self.param = param
        self.qf = qf
        self.setup(searcher, fieldname, text)

    def _score(self, weight, length):
        return intappscorer(weight, self.idf, self.cf, self.qf, self.dc, length, self.avgfl, self.param)


# ╔════════════════════════════════════════════════════════════╗
# ║  DO NOT MODIFY - BM25F baseline (참고용)                  ║
# ║  기본 제공 모델. QueryResult.py에서 scoring.BM25F()로 사용  ║
# ╚════════════════════════════════════════════════════════════╝

def bm25(idf, tf, fl, avgfl, B, K1):
    return idf * ((tf * (K1 + 1)) / (tf + K1 * ((1 - B) + B * fl / avgfl)))


class BM25F(WeightingModel):
    def __init__(self, B=0.75, K1=1.2, **kwargs):
        self.B = B
        self.K1 = K1

        self._field_B = {}
        for k, v in iteritems(kwargs):
            if k.endswith("_B"):
                fieldname = k[:-2]
                self._field_B[fieldname] = v

    def supports_block_quality(self):
        return True

    def scorer(self, searcher, fieldname, text, qf=1):
        if not searcher.schema[fieldname].scorable:
            return WeightScorer.for_(searcher, fieldname, text)

        if fieldname in self._field_B:
            B = self._field_B[fieldname]
        else:
            B = self.B

        return BM25FScorer(searcher, fieldname, text, B, self.K1, qf=qf)


class BM25FScorer(WeightLengthScorer):
    def __init__(self, searcher, fieldname, text, B, K1, qf=1):
        parent = searcher.get_parent()
        self.idf = parent.idf(fieldname, text)
        self.avgfl = parent.avg_field_length(fieldname) or 1

        self.B = B
        self.K1 = K1
        self.qf = qf
        self.setup(searcher, fieldname, text)

    def _score(self, weight, length):
        s = bm25(self.idf, weight, length, self.avgfl, self.B, self.K1)
        return s
