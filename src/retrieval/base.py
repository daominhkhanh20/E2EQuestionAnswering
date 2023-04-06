from typing import List, Optional, Union, Text
from abc import abstractmethod

from src.documents import Document


class BaseRetrieval:
    @abstractmethod
    def retrieval(self, query: str, top_k: int, **kwargs) -> List[Document]:
        raise NotImplementedError()

    def run(self, query: str, top_k: int, **kwargs):
        documents = self.retrieval(query=query, top_k=top_k, **kwargs)
        return {
            "query": query,
            "documents": documents,
            "top_k": top_k,
            **kwargs
        }
