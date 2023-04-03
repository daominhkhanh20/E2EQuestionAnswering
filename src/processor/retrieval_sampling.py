from typing import List, Optional, Dict, Text, Union
from src.documents import Corpus, Document

from sentence_transformers import SentenceTransformer, util


class TripleRetrievalSample:
    def __int__(self, document: str, question_positive: str, question_negative: str):
        self.document = document
        self.question_positive = question_positive
        self.question_negative = question_negative


class RetrievalGeneration:
    def __int__(self, list_retrieval_sample: List[TripleRetrievalSample], method_generation: str):
        self.list_retrieval_sample = list_retrieval_sample
        self.method_generation = method_generation

    @classmethod
    def random_generation(cls, corpus: Corpus, n_negative_sampling: int = 10):
        list_document_context = corpus.list_document_context
        for idx, document in enumerate(corpus.list_document):
            pass


    @classmethod
    def bm25_sampling(cls, corpus: Corpus, n_negative_sampling: int = 10):
        pass

    @classmethod
    def sentence_transformer_sampling(cls, corpus: Corpus, n_negative_sampling: int = 10,
                                      sentence_bert: SentenceTransformer = None):
        raise NotImplementedError()
