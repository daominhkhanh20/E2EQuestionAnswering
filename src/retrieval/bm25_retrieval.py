from abc import ABC
from typing import List, Dict
import numpy as np

from src.documents import Corpus, Document
from src.processor import BM25Scoring
from src.retrieval import BaseRetrieval


class BM25Retrieval(BaseRetrieval, ABC):
    def __init__(self, corpus: Corpus, tokenizer=None):
        super().__init__()
        self.list_document = corpus.list_document
        self.corpus = [context.lower().split(" ") for context in corpus.list_document_context
                       ]
        self.bm25_model = BM25Scoring(corpus=self.corpus, tokenizer=tokenizer)

    def retrieval(self, query: str, top_k: int, **kwargs) -> List[Document]:
        query = query.lower().split(" ")
        scores = self.bm25_model.get_scores(query)
        top_k_indexs = np.argsort(scores)[-top_k:]
        results = []
        for index in top_k_indexs:
            self.list_document[index].bm25_score = scores[index]
            results.append(self.list_document[index])
        return results
