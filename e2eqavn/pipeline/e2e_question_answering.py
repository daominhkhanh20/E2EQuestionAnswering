import logging
from typing import *

from .pipeline import Pipeline
from e2eqavn.retrieval import BaseRetrieval
from e2eqavn.mrc import BaseReader

logger = logging.getLogger(__name__)


class E2EQuestionAnswering(Pipeline):
    def __init__(self, retrieval: Union[BaseRetrieval, List[BaseRetrieval]],
                 reader: BaseReader):
        super().__init__()
        self.pipeline = Pipeline()
        if not isinstance(retrieval, List):
            self.pipeline.add_node(component=retrieval, name_component='Retrieval', input_component="Query")
        else:
            input_root = "Query"
            for idx, sub_retrieval in enumerate(retrieval):
                name = f"Retrieval_{idx}"
                self.pipeline.add_node(
                    component=sub_retrieval,
                    name_component=name,
                    input_component=input_root
                )
                input_root = name

    def run(self, query: str,
            top_k_bm25: int = 50,
            top_k_retrieval: int = 10,
            **kwargs):
        output = self.pipeline.run(
            query=query,
            top_k_bm25=top_k_bm25,
            top_k_retrieval=top_k_retrieval,
            **kwargs
        )
        return output
