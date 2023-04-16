from typing import *
import numpy as np
import torch
from numpy import array
from torch import Tensor
from tqdm import tqdm
import logging
from sentence_transformers import util
from sentence_transformers.evaluation import InformationRetrievalEvaluator

from .io import load_json_data
import hashlib

logger = logging.getLogger(__name__)


# def get_top_k_sample_for_sbert(query_embedding: Union[Tensor, np.array],
#                                corpus_embedding: Union[Tensor, np.array],
#                                top_k: int):
#     similarity_score = util.cos_sim(query_embedding, corpus_embedding)
#     scores, indexs = torch.topk(similarity_score, top_k, dim=1, largest=True, sorted=True)
#     return scores, indexs


def prepare_information_retrieval_evaluator(data: List[Dict], **kwargs) -> InformationRetrievalEvaluator:
    """
    :param data: List dictionary data
        Exammple:
            [
                {
                    "text": "xin chào bạn"
                    "qas": [
                        {
                            "question" : "question1",
                            "answers": [
                                {"text" : "answer1"},
                                {"text" : "answer2"}
                            ]
                        }
                    ]

                }
            ]
    :return:
    """
    logger.info(f"Start prepare evaluator for {len(data)} context")
    context_key = kwargs.get('context_key', 'context')
    qas_key = kwargs.get('qas_key', 'qas')
    question_key = kwargs.get('question_key', 'question')
    queries, corpus, relevant_docs = {}, {}, {}
    for sample in tqdm(data):
        context = sample[context_key]
        context_id = hashlib.sha1(str(context).encode('utf-8')).hexdigest()
        corpus[context_id] = context
        for ques in sample[qas_key]:
            question = ques[question_key]
            question_id = hashlib.sha1(str(question).encode('utf-8')).hexdigest()
            queries[question_id] = question
            if question_id not in relevant_docs:
                relevant_docs[question_id] = set()
            relevant_docs[question_id].add(context_id)
    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs
    )


def make_vnsquad_retrieval_evaluator(path_data_json: str, **kwargs):
    data = load_json_data(path_data_json)
    temp = []
    for context in data['data']:
        temp.extend(context['paragraphs'])
    return prepare_information_retrieval_evaluator(temp, **kwargs)
