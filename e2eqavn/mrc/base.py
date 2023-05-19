from typing import *
from abc import abstractmethod
from e2eqavn.documents import Document


class BaseReader:
    @abstractmethod
    def predict(self, query: Union[str, List[str]], documents: List[List[Document]], **kwargs):
        raise Exception("Not implemented")

    def run(self, query: Union[str, List[str]], documents: List[List[Document]], **kwargs):
        if isinstance(query, str):
            query = [query]
            
        if len(documents):
            return {
                "query": query,
                "answer": [],
                **kwargs
            }
        else:
            return {
                "query": query,
                "answer": self.predict(query, documents, **kwargs)
            }
