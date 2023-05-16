import re
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
from e2eqavn.utils.preprocess import process_text

logger = logging.getLogger(__name__)


def tokenize_function(example, tokenizer, max_length: int = 368):
    # print(example)
    example["question"] = example["question"].split()
    example["context"] = example["context"].split()
    max_len_single_sentence = 368

    question_sub_words_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(w)) for w in example["question"]]
    context_sub_words_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(w)) for w in example["context"]]
    valid = True
    if len([j for i in question_sub_words_ids + context_sub_words_ids for j in
            i]) > max_len_single_sentence - 1:
        question_ids = [j for i in question_sub_words_ids for j in i]
        context_ids = [j for i in context_sub_words_ids[:example['answer_word_end_idx'] + 1] for j in i]
        remain_tokens = max_len_single_sentence - 1 - len(question_ids)
        if len(question_ids + context_ids) < max_len_single_sentence - 1:
            context_sub_words_ids_revise = context_sub_words_ids[:example['answer_word_end_idx'] + 1]
            idx = example['answer_word_end_idx'] + 1
            while len([j for i in (context_sub_words_ids_revise + [context_sub_words_ids[idx]]) for j in
                       i]) < remain_tokens and idx < len(context_sub_words_ids):
                context_sub_words_ids_revise.append(context_sub_words_ids[idx])
                idx += 1
            context_sub_words_ids = context_sub_words_ids_revise
        else:
            valid = False

    question_sub_words_ids = [[tokenizer.bos_token_id]] + question_sub_words_ids + [[tokenizer.eos_token_id]]
    context_sub_words_ids = context_sub_words_ids + [[tokenizer.eos_token_id]]

    input_ids = [j for i in question_sub_words_ids + context_sub_words_ids for j in i]
    if len(input_ids) > max_len_single_sentence + 2:
        valid = False

    words_lengths = [len(item) for item in question_sub_words_ids + context_sub_words_ids]
    attention_mask = [1] * len(input_ids)
    if len(example[ANSWER]) == 0:
        start_idx = 0
        end_idx = 0
    else:
        start_idx = example['answer_word_start_idx'] + len(question_sub_words_ids)
        end_idx = example['answer_word_end_idx'] + len(question_sub_words_ids)
    return {
        INPUT_IDS: input_ids,
        ATTENTION_MASK: attention_mask,
        WORDS_LENGTH: words_lengths,
        START_IDX: start_idx,
        END_IDX: end_idx,
        "is_valid": valid
    }


def calculate_input_training_for_qav2(example: dict, tokenizer, max_length: int):
    # question_ids context_ids
    original_max_length = max_length
    context = example[CONTEXT]
    question = example[QUESTION]
    answer_start_idx = example[ANSWER_WORD_START_IDX]
    answer_end_idx = example[ANSWER_WORD_END_IDX]
    context_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word)) for word in context.split()]
    question_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word)) for word in question.split()]
    arr_size_sub_word_context_ids = [len(sub_ids) for sub_ids in context_ids]
    arr_size_sub_word_question_ids = [len(sub_ids) for sub_ids in question_ids]
    if not example[IS_VALID] or sum(arr_size_sub_word_context_ids) > 300 or sum(arr_size_sub_word_question_ids) > 100:
        return {
            INPUT_IDS: None,
            ATTENTION_MASK: None,
            START_IDX: 0,
            END_IDX: 0,
            WORDS_LENGTH: None,
            'is_valid': False
        }

    is_valid = True
    if sum(arr_size_sub_word_question_ids) + sum(arr_size_sub_word_context_ids) > max_length - 5:
        if sum(arr_size_sub_word_question_ids) + sum(
                arr_size_sub_word_context_ids[:answer_end_idx + 1]) > max_length - 5:
            is_valid = False
        else:
            current_length = sum(arr_size_sub_word_question_ids) + sum(
                arr_size_sub_word_context_ids[: answer_end_idx + 1]) + 3  # for 3 special token
            tmp = answer_end_idx + 1
            while current_length + arr_size_sub_word_context_ids[tmp] < max_length and tmp < len(
                    arr_size_sub_word_context_ids) - 1:
                current_length += arr_size_sub_word_context_ids[tmp]
                tmp += 1
            context_ids = context_ids[: tmp]
    if tokenizer.bos_token_id is not None and tokenizer.eos_token_id is not None:
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id
    else:
        temp_sample = tokenizer('xin chào')
        bos_token_id = temp_sample['input_ids'][0]
        eos_token_id = temp_sample['input_ids'][-1]

    question_final_ids = [[bos_token_id]] + question_ids + [[eos_token_id]]
    context_final_ids = context_ids + [[eos_token_id]]
    input_ids = [id for sub_ids in question_final_ids + context_final_ids for id in sub_ids]
    words_length = [len(item) for item in question_final_ids + context_final_ids]
    if len(input_ids) > original_max_length:
        is_valid = False

    attention_mask = [1] * len(input_ids)

    return {
        INPUT_IDS: input_ids,
        ATTENTION_MASK: attention_mask,
        START_IDX: answer_start_idx + len(question_final_ids) if len(example[ANSWER]) > 0 else 0,
        END_IDX: answer_end_idx + len(question_final_ids) if len(example[ANSWER]) > 0 else 0,
        WORDS_LENGTH: words_length,
        'is_valid': is_valid
    }


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
