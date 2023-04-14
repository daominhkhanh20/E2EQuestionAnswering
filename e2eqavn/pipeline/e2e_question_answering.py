import logging
from typing import *

from .pipeline import Pipeline
from e2eqavn.retrieval import BaseRetrieval, SBertRetrieval
from e2eqavn.mrc import BaseReader

logger = logging.getLogger(__name__)


class E2EQuestionAnsweringPipeline(Pipeline):
    def __init__(self, retrieval: SBertRetrieval,
                 reader: BaseReader = None):
        super().__init__()
        self.pipeline = Pipeline()
        self.retrieval = retrieval
        # if not isinstance(retrieval, List):
        #     self.pipeline.add_node(component=retrieval, name_component='Retrieval', input_component="root")
        # else:
        #     input_root = "root"
        #     for idx, sub_retrieval in enumerate(retrieval):
        #         name = f"Retrieval_{idx}"
        #         self.pipeline.add_node(
        #             component=sub_retrieval,
        #             name_component=name,
        #             input_component=input_root
        #         )
        #         input_root = name

    def run(self, queries: Union[str, List[str]],
            top_k_bm25: int = 50,
            top_k_sbert: int = 10,
            **kwargs):
        if isinstance(queries, str):
            queries = [queries]

        # output = self.pipeline.run(
        #     queries=queries,
        #     top_k_bm25=top_k_bm25,
        #     top_k_sbert=top_k_sbert,
        #     **kwargs
        # )
        outs = self.retrieval.retrieval(
            queries,
            top_k=10,
            top_k_sbert=top_k_sbert
        )
        return outs
