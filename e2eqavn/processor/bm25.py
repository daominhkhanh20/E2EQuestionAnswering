from abc import ABC
from typing import List, Optional, Union, Text
from multiprocessing import Pool, cpu_count
import numpy as np
import math


class BM25Base:
    def __init__(self, corpus: Union[List[str], List[List[str]]]):
        self.corpus = corpus
        self.n_docs = len(self.corpus)
        self.avg_document_length = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        if isinstance(corpus[0], str):
            corpus = [doc.lower().split(" ") for doc in corpus]

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus: List[str]):
        nd = {}
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)
            for word, freq in frequencies.items():
                if word in nd:
                    nd[word] += 1
                else:
                    nd[word] = 1
        self.avg_document_length = num_doc / self.n_docs
        return nd

    # def _tokenizer_corpus(self, corpus: List[str]):
    #     pool = Pool(cpu_count())
    #     tokenized_corpus = pool.map(self.tokenizer, corpus)
    #     return tokenized_corpus

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        raise NotImplementedError()

    # def get_top_n(self, query, documents, n=5):
    #
    #     assert self.n_docs == len(documents), "The documents given don't match the index corpus!"
    #
    #     scores = self.get_scores(query)
    #     top_n = np.argsort(scores)[::-1][:n]
    #     return [documents[i] for i in top_n]


class BM25Scoring(BM25Base, ABC):
    def __init__(self, corpus: Union[List[str], List[List[str]]], k1=1.5, b=0.75, epsilon=0.25, alpha=2,
                 penalty_oov=False):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.alpha = alpha
        self.penalty_oov = penalty_oov
        super().__init__(corpus)

    def _calc_idf(self, nd):
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.n_docs - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query: Union[List, str]):
        if isinstance(query, str):
            query = query.lower().split(" ")
        score = np.zeros(self.n_docs)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            if self.penalty_oov:
                if self.idf.get(q) is not None:
                    score += self.idf.get(q) * (
                            q_freq * (self.k1 + 1) / (
                                q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_document_length)))
                else:
                    temp = self.alpha * (self.k1 + 1) / (
                                0 + self.k1 * (1 - self.b + self.b * doc_len / self.avg_document_length))
                    score -= temp
            else:
                score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                                   (q_freq + self.k1 * (
                                                               1 - self.b + self.b * doc_len / self.avg_document_length)))
        return score

    def get_top_k(self, query: Union[List[str], str], top_k: int):
        if isinstance(query, str):
            query = query.lower().split(" ")
        scores = self.get_scores(query)
        top_k_idxs = np.argsort(scores)[-top_k:]
        return {idx: scores[idx] for idx in top_k_idxs}
