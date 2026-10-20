"""
Minimal BM25 implementation with no external dependencies.

NOTE: In production (with internet access) this would simply be:
    from rank_bm25 import BM25Okapi
This hand-rolled version exists so the full pipeline is runnable
in an offline/sandboxed environment without pip access, while
preserving identical retrieval behavior/interface.
"""
import math
import re
from collections import Counter


def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus_tokens
        self.doc_lens = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.doc_lens) / len(corpus_tokens)
        self.doc_freqs = [Counter(doc) for doc in corpus_tokens]
        self.n_docs = len(corpus_tokens)
        self.idf = self._compute_idf()

    def _compute_idf(self):
        df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                df[term] += 1
        idf = {}
        for term, freq in df.items():
            idf[term] = math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
        return idf

    def score(self, query_tokens, doc_index):
        score = 0.0
        doc_freq = self.doc_freqs[doc_index]
        doc_len = self.doc_lens[doc_index]
        for term in query_tokens:
            if term not in doc_freq:
                continue
            freq = doc_freq[term]
            idf = self.idf.get(term, 0.0)
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numerator / denominator
        return score

    def get_scores(self, query: str):
        query_tokens = tokenize(query)
        return [self.score(query_tokens, i) for i in range(self.n_docs)]
