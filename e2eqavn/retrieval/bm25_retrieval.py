from abc import ABC
from typing import List, Dict
import numpy as np

from e2eqavn.documents import Corpus, Document
from e2eqavn.processor import BM25Scoring
from e2eqavn.retrieval import BaseRetrieval


class BM25Retrieval(BaseRetrieval, ABC):
    def __init__(self, corpus: Corpus):
        super().__init__()
        self.list_document = corpus.list_document
        self.bm25_model = BM25Scoring(corpus=corpus.list_document_context)

    def retrieval(self, query: str, top_k: int = 10, **kwargs) -> List[Document]:
        if kwargs.get("top_k_bm25", None):
            top_k = kwargs.get("top_k_bm25")
        else:
            top_k = top_k

        query = query.lower().split(" ")
        scores = self.bm25_model.get_scores(query)
        top_k_indexs = np.argsort(scores)[-top_k:]
        results = []
        for index in top_k_indexs:
            self.list_document[index].bm25_score = scores[index]
            results.append(self.list_document[index])
        return results
