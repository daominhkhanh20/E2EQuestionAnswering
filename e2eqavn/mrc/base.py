from typing import *
from abc import abstractmethod
from e2eqavn.documents import Document


class BaseReader:
    @abstractmethod
    def predict(self, queries: Union[str, List[str]], documents: List[List[Document]], **kwargs):
        raise Exception("Not implemented")

    def run(self, queries: Union[str, List[str]], documents: List[List[Document]], **kwargs):
        if isinstance(queries, str):
            queries = [queries]
            
        if len(documents) == 0:
            return {
                "query": queries,
                "answer": [],
                **kwargs
            }
        else:
            return {
                "query": queries,
                "documents": [[doc.__dict__ for doc in list_document] for list_document in documents],
                "answer": self.predict(queries, documents, **kwargs)
            }
