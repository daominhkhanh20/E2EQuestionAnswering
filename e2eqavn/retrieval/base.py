from typing import List, Optional, Union, Text
from abc import abstractmethod

from e2eqavn.documents import Document


class BaseRetrieval:
    @abstractmethod
    def retrieval(self, queries: List[str], top_k: int, **kwargs) -> List[List[Document]]:
        raise NotImplementedError()

    def run(self, queries: List[str], top_k: int = 10, **kwargs):
        if 'documents' in kwargs:
            kwargs.pop('documents')
        documents = self.retrieval(queries=queries, top_k=top_k, **kwargs)
        # for tmp_docs in documents:
        #     print([doc.index for doc in tmp_docs])
        #     print([doc.embedding_similarity_score for doc in tmp_docs])
        return {
            "queries": queries,
            "documents": documents,
            "top_k": top_k,
            **kwargs
        }
