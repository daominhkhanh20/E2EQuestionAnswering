from typing import *
import numpy as np
import torch
from numpy import array
from torch import Tensor
from tqdm import tqdm
import logging
from torch import Tensor
from transformers import AutoTokenizer
from sentence_transformers import util
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from .io import load_json_data
import hashlib

from e2eqavn.keywords import *

logger = logging.getLogger(__name__)


def calculate_input_training_for_qa(example, tokenizer, is_document_right: bool):
    context = example[CONTEXT]
    question = example[QUESTION]
    answer = example[ANSWER]
    output_tokenizer_samples = tokenizer(
        context if is_document_right else question,
        question if is_document_right else context,
        return_offsets_mapping=True
    )
    cls_token_id = tokenizer.cls_token_id
    input_ids = output_tokenizer_samples[INPUT_IDS]
    mask = output_tokenizer_samples[ATTENTION_MASK]
    cls_index = input_ids.index(cls_token_id)
    offset_mapping = output_tokenizer_samples[OFFSET_MAPPING]
    data = {
        INPUT_IDS: input_ids,
        ATTENTION_MASK: mask
    }
    try:
        start_index = context.index(answer)
        end_index = start_index + len(answer)
        token_start_index, token_end_index = -1, -1
        flag_start_index, flag_end_index = False, False
        i = 0
        while not flag_start_index and i < len(offset_mapping):
            if offset_mapping[i][0] == start_index:
                token_start_index = i
                flag_start_index = True
            i += 1

        i = len(input_ids) - 1
        while not flag_end_index and i > -1:
            if offset_mapping[i][1] == end_index:
                token_end_index = i
                flag_end_index = True
            i -= 1
        if token_end_index > -1 and token_start_index > -1:
            data[START_IDX] = start_index
            data[END_IDX] = end_index
        else:
            data[START_IDX] = cls_index
            data[END_IDX] = cls_index
            logger.info(f"""
            Answer: '{answer}' \n
            Not found in document context: {context}
            """)
        return data
    except Exception as e:
        logger.info(e)
        data[START_IDX] = cls_index
        data[END_IDX] = cls_index
        logger.info(f"""
        Answer: '{answer}' \n
        Not found in document context: {context}
        """)
        return data


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
