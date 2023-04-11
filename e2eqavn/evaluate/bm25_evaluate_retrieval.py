from typing import *
from sentence_transformers.evaluation import SentenceEvaluator

from e2eqavn.retrieval import BaseRetrieval


class InformationRetrievalEvaluatorCustom(SentenceEvaluator):
    def __init__(self, retrieval: Union[BaseRetrieval, List[BaseRetrieval]]):
        pass