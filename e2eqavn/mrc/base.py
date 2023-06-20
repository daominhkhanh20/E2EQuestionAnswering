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
            predict, raw_predict = self.predict(queries, documents, **kwargs)
            return {
                "query": queries,
                "documents": documents,
                "answer": predict,
                'reader_logging': raw_predict
            }
