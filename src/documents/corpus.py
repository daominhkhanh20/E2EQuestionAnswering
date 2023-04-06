from typing import List, Dict, Optional, Text, Union
from uuid import uuid4
import numpy as np
import math
import logging
from torch import Tensor
from collections import defaultdict
from src.utils.io import load_json_data, write_json_file

logger = logging.getLogger(__name__)


# class AnswerInformation:
#     def __init__(self, document_context: str, question: str, answer: str,
#                  answer_start_idx: int = None, answer_end_idx: int = None):
#         self.document_context = document_context
#         self.question = question
#         self.answer = answer
#         self.answer_start_idx = answer_start_idx
#         self.answer_end_idx = answer_end_idx
#
#     def find_index_answer(self):
#         """
#         Method find span answer index base on tokenizer for Machine Reading Comprehension task
#         :return: answer_start and answer_end
#         """
#         raise NotImplementedError()


class PairQuestionAnswers:
    def __init__(self, document_id: str, document_context: str, question: str, list_answers: List[str]):
        self.document_id = document_id
        self.document_context = document_context
        self.question = question
        self.list_answers = list_answers

    def find_index_answer(self):
        """
        Method find span answer index base on tokenizer for Machine Reading Comprehension task
        :return: answer_start and answer_end
        """
        raise NotImplementedError()

    # @classmethod
    # def init_class(cls, document_id, document_context: str,
    #                question: str, answers: List[str]):
    #     tmp_list = []
    #     for answer in answers:
    #         tmp_list.append(
    #             AnswerInformation(
    #                 document_context=document_context,
    #                 question=question,
    #                 answer=answer
    #             )
    #         )
    #     return cls(document_id, document_context, tmp_list)


class Document:
    def __init__(self, document_context: str, document_id: str = None,
                 list_pair_question_answers: List[PairQuestionAnswers] = None,
                 bm25_score: float = 0,
                 embedding: Union[np.array, Tensor] = None):
        self.document_context = document_context
        self.bm25_score = bm25_score
        if document_id:
            self.document_id = str(uuid4())
        self.embedding = embedding
        self.list_pair_question_answers = list_pair_question_answers

    @classmethod
    def init_document(cls, document_id: str, document_context: str,
                      dict_question_answers: Dict[str, List]):
        """
        :param document_id:
        :param document_context:
        :param dict_question_answers:
            example: {
                "question1": [
                    "answer1",
                    "answer2"
                ],
                ....
            }
        :return:
        """
        temp = []
        if len(dict_question_answers) > 0:
            for question, list_answer in dict_question_answers.items():
                temp.append(
                    PairQuestionAnswers(
                        document_id=document_id,
                        document_context=document_context,
                        question=question,
                        list_answers=list_answer
                    )
                )
            return cls(document_context=document_context, document_id=document_id, list_pair_question_answers=temp)
        return cls(document_context=document_context, document_id=document_id, list_pair_question_answers=[])


class Corpus:
    def __init__(self, list_document: List[Document]):
        self.list_document = list_document
        self.n_document = len(self.list_document)
        self.n_pair_question_answer = 0
        self.list_document_context = [document.document_context for document in list_document]
        for document in self.list_document:
            self.n_pair_question_answer += len(document.list_pair_question_answers)

    @classmethod
    def chunk_document(cls, context: str, max_length: int = 400, overlapping_size: int = 30):
        size = max_length - overlapping_size
        list_words = context.split(" ")
        n_chunk = math.ceil(len(list_words) / size)
        list_context = []
        for i in range(n_chunk):
            temp_context = " ".join(list_words[i * size: i * size + max_length])
            list_context.append(temp_context)
        return list_context

    @classmethod
    def parser_uit_squad(cls, path_file: str, mode_chunking: bool = False,
                         max_length: int = 400, overlapping_size: int = 30):
        data = load_json_data(path_file)
        list_document = []
        for context in data['data']:
            for paragraph in context['paragraphs']:
                document_context = paragraph['context']
                if not mode_chunking:
                    document_id = paragraph.get('document_id', str(uuid4()))
                    dict_question_answers = defaultdict(list)
                    for question in paragraph['qas']:
                        for answer in question['answers']:
                            dict_question_answers[question['question']].append(answer['text'])

                    list_document.append(
                        Document.init_document(
                            document_id=document_id,
                            document_context=document_context,
                            dict_question_answers=dict_question_answers
                        )
                    )
                else:
                    list_context = cls.chunk_document(document_context, max_length, overlapping_size)
                    list_context_id = [str(uuid4()) for i in range(len(list_context))]
                    dict_question_answers = {key: {} for key in list_context_id}
                    for question in paragraph['qas']:
                        for answer in question['answers']:
                            flag_exist = False
                            for idx, context_chunk in enumerate(list_context):
                                if answer['text'] in context_chunk:
                                    if question['question'] not in dict_question_answers[list_context_id[idx]]:
                                        dict_question_answers[list_context_id[idx]][question['question']] = [
                                            answer['text']]
                                    else:
                                        dict_question_answers[list_context_id[idx]][question['question']].append(
                                            answer['text'])
                                    flag_exist = True
                                    break
                            if not flag_exist:
                                logger.info(f"Answer: {answer['text']} \n "
                                            f"N chunk context: {len(list_context)}\n"
                                            f"List Context: {list_context} \n"
                                            f"Answer doesn't exist in list context\n\n")

                    for idx, (key, value) in enumerate(dict_question_answers.items()):
                        list_document.append(
                            Document.init_document(
                                document_id=key,
                                document_context=list_context[idx],
                                dict_question_answers=dict_question_answers[list_context_id[idx]]
                            )
                        )

        return cls(list_document=list_document)

    def save_corpus(self, path_file: str):
        infor = {}
        for document in self.list_document:
            infor[document.document_id] = {
                "context": document.document_context,
                "qas": []
            }
            for question_answer in document.list_pair_question_answers:
                infor[document.document_id]['qas'].append(
                    {
                        "question": question_answer.question,
                        "answers": question_answer.list_answers
                    }
                )
        write_json_file(infor, path_file)

