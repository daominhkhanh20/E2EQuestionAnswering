import logging
from typing import *

from .pipeline import Pipeline
from e2eqavn.utils.preprocess import process_text
from e2eqavn.retrieval import BaseRetrieval
from e2eqavn.mrc import BaseReader

logger = logging.getLogger(__name__)


class E2EQuestionAnsweringPipeline(Pipeline):
    def __init__(self, retrieval: Union[BaseRetrieval, List[BaseRetrieval]] = None,
                 reader: BaseReader = None):
        super().__init__()
        self.pipeline = Pipeline()
        if retrieval is not None:
            if not isinstance(retrieval, List):
                self.pipeline.add_node(component=retrieval, name_component='Retrieval', input_component="root")
            else:
                input_root = "root"
                for idx, sub_retrieval in enumerate(retrieval):
                    name = f"Retrieval_{idx}"
                    self.pipeline.add_node(
                        component=sub_retrieval,
                        name_component=name,
                        input_component=input_root
                    )
                    input_root = name
        if reader is not None:
            self.pipeline.add_node(
                component=reader,
                name_component='reader',
                input_component=input_root
            )

    def run(self, queries: Union[str, List[str]],
            top_k_bm25: int = 50,
            top_k_sbert: int = 10,
            **kwargs):
        if isinstance(queries, str):
            queries = [queries]
        queries = [process_text(query) for query in queries]
        
        output = self.pipeline.run(
            queries=queries,
            top_k_bm25=top_k_bm25,
            top_k_sbert=top_k_sbert,
            **kwargs
        )
        return output
