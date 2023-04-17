from typing import List, Optional, Dict, Text, Union
import torch.cuda
import numpy as np
import random
from random import sample
from tqdm import tqdm
import logging

from e2eqavn.processor.bm25 import BM25Scoring
from e2eqavn.documents import Corpus, Document
from e2eqavn.keywords import *
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


class TripleRetrievalSample:
    def __init__(self, question: str, document_positive: str, document_negative: str):
        self.question = question
        self.document_positive = document_positive
        self.document_negative = document_negative


class PairRetrievalSample:
    def __init__(self, question: str, document: str, label: int):
        self.question = question
        self.document = document
        self.label = label


class RetrievalGeneration:
    method_generation: str = 'random'
    method_train: str = 'triplet'

    def __init__(self, list_retrieval_sample: Union[List[TripleRetrievalSample], List[PairRetrievalSample]],
                 method_generation: str, method_train: str, **kwargs):
        self.list_retrieval_sample = list_retrieval_sample
        self.method_generation = method_generation
        self.method_train = method_train
        self.__dict__.update(kwargs)

    @classmethod
    def generate_sampling(cls, corpus: Corpus, **kwargs):
        method_generation = kwargs.get('method_generation', cls.method_generation)
        method_train = kwargs.get('method_train', cls.method_train)
        n_negative = kwargs.get('n_negative', 5)
        list_retrieval_sample = []
        list_document_context = corpus.list_document_context
        list_context_index = [i for i in range(len(corpus.list_document_context))]

        if method_generation.lower() not in [RANDOM, BM25, SBERT]:
            raise Exception(f"Method generation '{method_generation}' isn't support")

        if method_train.lower() not in [TRIPLET, PAIR]:
            raise Exception(f"Method training '{method_train}' isn't support")

        logger.info(f"Start generate retrieval sample with method: {method_generation}")
        if method_generation == 'sbert':
            sbert_model = SentenceTransformer(kwargs.get('sbert_model_gen', 'keepitreal/vietnamese-sbert'))
            logger.info("Start encode corpus embedding")
            corpus_embedding = sbert_model.encode(
                sentences=list_document_context,
                batch_size=kwargs.get('batch_size_encode', 32),
                show_progress_bar=True,
                convert_to_numpy=True,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        elif method_generation == 'bm25':
            bm25_model = BM25Scoring(
                corpus=list_document_context
            )

        for idx, document in tqdm(enumerate(corpus.list_document), total=len(corpus.list_document)):
            for list_pair_qa in document.list_pair_question_answers:
                question = list_pair_qa.question
                if method_generation == 'random':
                    list_negative_docs_index = cls.random_generation(
                        list_context_index=list_context_index,
                        n_negative=n_negative,
                    )
                elif method_generation == 'bm25':
                    list_negative_docs_index = cls.bm25_generation(
                        bm25_model=bm25_model,
                        query=question,
                        n_negative=n_negative
                    )
                elif method_generation == 'sbert':
                    query_embedding = sbert_model.encode(question, show_progress_bar=False)
                    list_negative_docs_index = cls.sentence_transformer_generation(
                        query_embedding=query_embedding,
                        corpus_embedding=corpus_embedding,
                        n_negative=n_negative
                    )
                else:
                    raise Exception("List negative index is None")
                if idx in list_negative_docs_index:
                    list_negative_docs_index.remove(idx)
                if method_train == 'triplet':
                    for neg_idx in list_negative_docs_index:
                        list_retrieval_sample.append(
                            TripleRetrievalSample(
                                question=question,
                                document_positive=document.document_context,
                                document_negative=list_document_context[neg_idx]
                            )
                        )
                elif method_train == 'pair':
                    list_retrieval_sample.append(
                        PairRetrievalSample(
                            question=question,
                            document=document.document_context,
                            label=1
                        )
                    )
                    for neg_idx in list_negative_docs_index:
                        list_retrieval_sample.append(
                            PairRetrievalSample(
                                question=question,
                                document=list_document_context[neg_idx],
                                label=0
                            )
                        )

        return cls(list_retrieval_sample, **kwargs)

    @classmethod
    def random_generation(cls, list_context_index: List[int], n_negative: int):
        return list(random.sample(list_context_index, n_negative))

    @classmethod
    def bm25_generation(cls, bm25_model: BM25Scoring, query: str, n_negative: int):
        scores = bm25_model.get_scores(query).reshape(-1)
        sorted_idxs = np.argsort(scores)
        sub_haft_best = int(n_negative/2)
        top_index = list(sorted_idxs[-sub_haft_best:])
        random_index = list(np.random.choice(sorted_idxs[:-sub_haft_best], n_negative - sub_haft_best))
        return random_index + top_index

    @classmethod
    def sentence_transformer_generation(cls, corpus_embedding, query_embedding, n_negative: int, **kwargs):
        sim_score = util.cos_sim(query_embedding, corpus_embedding).cpu().numpy().reshape(-1)
        sort_index = np.argsort(sim_score)
        sub_haft = int(n_negative/2)
        top_index = list(sort_index[-sub_haft:])
        random_index = list(sample(sort_index[:-sub_haft], n_negative - sub_haft))
        return top_index + random_index

