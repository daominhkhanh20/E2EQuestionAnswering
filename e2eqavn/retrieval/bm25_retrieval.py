from abc import ABC
from typing import List, Dict, Union
import numpy as np
import multiprocessing as mp
from copy import deepcopy
from multiprocessing import Pool

from e2eqavn.documents import Corpus, Document
from e2eqavn.processor import BM25Scoring
from e2eqavn.retrieval import BaseRetrieval


class BM25Retrieval(BaseRetrieval, ABC):
    def __init__(self, corpus: Corpus):
        super().__init__()
        self.list_documents = corpus.list_document
        self.bm25_model = BM25Scoring(corpus=corpus.list_document_context)

    def retrieval(self, queries: List[str], top_k: int = 10, **kwargs) -> List[List[Document]]:
        if kwargs.get("top_k_bm25", None):
            top_k = kwargs.get("top_k_bm25")

        args = [(query, top_k) for query in queries]
        list_docs = []
        with Pool(processes=mp.cpu_count()) as pool:
            for mapping_idx_score in pool.starmap(self.bm25_model.get_top_k, args):
                tmp = []
                max_score = max(mapping_idx_score.values())
                for idx in mapping_idx_score.keys():
                    document = Document(
                        index=self.list_documents[idx].index,
                        document_id=self.list_documents[idx].document_id,
                        document_context=self.list_documents[idx].document_context,
                        bm25_score=mapping_idx_score[idx] / max_score
                    )
                    tmp.append(document)
                list_docs.append(tmp)

        # query = query.lower().split(" ")
        # mapping_idx_score = self.bm25_model.get_top_k(query=query, top_k=top_k)
        # results = []
        # for index in mapping_idx_score.keys():
        #     self.list_document[index].bm25_score = mapping_idx_score[index]
        #     results.append(self.list_document[index])
        return list_docs

    # def batch_retrieval(self, queries: List[str], top_k: int = 10, **kwargs):
    #     if kwargs.get("top_k_bm25", None):
    #         top_k = kwargs.get("top_k_bm25")
    #     else:
    #         top_k = top_k
    #     list_docs = []
    #     args = [(query, top_k) for query in queries]
    #     with Pool(processes=mp.cpu_count()) as pool:
    #         for result in pool.starmap(self.retrieval, args):
    #             list_docs.append(result)
    #     if kwargs.get('return_index', None):
    #         return [[doc.index for doc in result] for result in list_docs]
    #     elif kwargs.get('return_id', None):
    #         return [[doc.document_id for doc in result] for result in list_docs]
    #     return list_docs
