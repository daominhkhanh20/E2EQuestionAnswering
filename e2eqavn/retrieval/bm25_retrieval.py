from abc import ABC
from typing import List, Dict, Union
import numpy as np
import multiprocessing as mp
from copy import deepcopy
from multiprocessing import Pool
from tqdm import tqdm
import logging

from e2eqavn.documents import Corpus, Document
from e2eqavn.processor import BM25Scoring
from e2eqavn.retrieval import BaseRetrieval

logger = logging.getLogger(__name__)


class BM25Retrieval(BaseRetrieval, ABC):
    def __init__(self, corpus: Corpus):
        super().__init__()
        self.list_documents = corpus.list_document
        self.bm25_model = BM25Scoring(corpus=corpus.list_document_context)

    def retrieval(self, queries: Union[List[str], str], top_k: int = 10, **kwargs) -> List[List[Document]]:
        if isinstance(queries, str):
            queries = [queries]
        if kwargs.get("top_k_bm25", None):
            top_k = kwargs.get("top_k_bm25")
        if len(queries) == 1:
            mapping_idx_score = self.bm25_model.get_top_k(queries[0], top_k)
            list_docs = []
            max_score = max(mapping_idx_score.values())
            for idx in mapping_idx_score.keys():
                document = Document(
                    index=self.list_documents[idx].index,
                    document_id=self.list_documents[idx].document_id,
                    document_context=self.list_documents[idx].document_context,
                    bm25_score=mapping_idx_score[idx] / max_score,
                    score=mapping_idx_score[idx] / 2
                )
                list_docs.append(document)
            list_docs = [list_docs]
        else:
            args = [(query, top_k) for query in queries]
            list_docs = []
            with Pool(processes=mp.cpu_count()) as pool:
                for mapping_idx_score in tqdm(pool.starmap(self.bm25_model.get_top_k, args)):
                    tmp = []
                    max_score = max(mapping_idx_score.values())
                    for idx in mapping_idx_score.keys():
                        document = Document(
                            index=self.list_documents[idx].index,
                            document_id=self.list_documents[idx].document_id,
                            document_context=self.list_documents[idx].document_context,
                            bm25_score=mapping_idx_score[idx] / max_score,
                            score=mapping_idx_score[idx] / 2
                        )
                        tmp.append(document)
                    list_docs.append(tmp)

        # logger.info(f"Result BM25: {len(list_docs)}")
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
