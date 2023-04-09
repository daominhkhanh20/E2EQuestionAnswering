from typing import List, Dict, Optional, Text, Union
from uuid import uuid4
# from pyvi import ViTokenizer
import numpy as np
import math
import logging
import hashlib
from torch import Tensor
from collections import defaultdict
from e2eqavn.utils.io import load_json_data, write_json_file

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

    # def find_index_answer(self):
    #     """
    #     Method find span answer index base on tokenizer for Machine Reading Comprehension task
    #     :return: answer_start and answer_end
    #     """
    #     raise NotImplementedError()

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
                 embedding_similarity_score: float = 0,
                 embedding: Union[np.array, Tensor] = None,
                 index: int = 0,
                 pyvi_mode: bool = False):
        self.document_context = document_context
        self.index = index
        # if pyvi_mode:
        #     self.document_context = ViTokenizer.tokenize(self.document_context)
        self.bm25_score = bm25_score
        self.embedding_similarity_score = embedding_similarity_score
        if document_id:
            self.document_id = hashlib.sha1(str(self.document_context).encode('utf-8')).hexdigest()
        self.embedding = embedding
        self.list_pair_question_answers = list_pair_question_answers

    @classmethod
    def init_document(cls, document_id: str, document_context: str,
                      dict_question_answers: Dict[str, List], index: int):
        """
        :param index:
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
            return cls(document_context=document_context, document_id=document_id, list_pair_question_answers=temp, index=index)
        return cls(document_context=document_context, document_id=document_id, list_pair_question_answers=[], index=index)


class Corpus:
    context_key: str = 'context'
    qas_key: str = 'qas'
    question_key: str = 'question'
    answers_key: str = 'answers'
    answer_key: str = 'text'
    max_length: int = 400
    overlapping_size: int = 40
    doc_th = 0

    def __init__(self, list_document: List[Document], **kwargs):
        self.list_document = list_document
        self.n_document = len(self.list_document)
        self.n_pair_question_answer = 0
        self.list_document_context = [document.document_context for document in list_document]
        for document in self.list_document:
            self.n_pair_question_answer += len(document.list_pair_question_answers)
        self.__dict__.update(kwargs)

    @classmethod
    def get_documents(cls, context: Dict, **kwargs):
        context_key = kwargs.get('context_key', cls.context_key)
        qas_key = kwargs.get('qas_key', cls.qas_key)
        question_key = kwargs.get('question_key', cls.question_key)
        answers_key = kwargs.get('answers_key', cls.answers_key)
        answer_key = kwargs.get('answer_key', cls.answer_key)
        list_document = []
        document_context = context[context_key]
        if kwargs.get('mode_chunking', False):
            document_id = hashlib.sha1(str(document_context).encode('utf-8')).hexdigest()
            dict_question_answers = defaultdict(list)
            for question in context[qas_key]:
                for answer in question[answers_key]:
                    dict_question_answers[question[question_key]].append(answer[answer_key])

            list_document.append(
                Document.init_document(
                    document_id=document_id,
                    document_context=document_context,
                    dict_question_answers=dict_question_answers,
                    index=cls.doc_th
                )
            )
            cls.doc_th += 1
        else:
            list_context = cls.chunk_document(document_context, **kwargs)
            list_context_id = [hashlib.sha1(str(context).encode('utf-8')).hexdigest()
                               for context in list_context]
            dict_question_answers = {key: {} for key in list_context_id}
            for question in context[qas_key]:
                for answer in question[answers_key]:
                    flag_exist = False
                    for idx, context_chunk in enumerate(list_context):
                        if answer[answer_key] in context_chunk:
                            if question[question_key] not in dict_question_answers[list_context_id[idx]]:
                                dict_question_answers[list_context_id[idx]][question[question_key]] = [
                                    answer[answer_key]]
                            else:
                                dict_question_answers[list_context_id[idx]][question[question_key]].append(
                                    answer[answer_key])
                            flag_exist = True
                            break
                    if not flag_exist:
                        logger.info(f"Answer: {answer[answer_key]} \n "
                                    f"N chunk context: {len(list_context)}\n"
                                    f"List Context: {list_context} \n"
                                    f"Answer doesn't exist in context\n\n")

            for idx, (key, value) in enumerate(dict_question_answers.items()):
                list_document.append(
                    Document.init_document(
                        document_id=key,
                        document_context=list_context[idx],
                        dict_question_answers=dict_question_answers[list_context_id[idx]],
                        index=cls.doc_th
                    )
                )
                cls.doc_th += 1
        return list_document

    @classmethod
    def init_corpus(cls, corpus: List[Dict], **kwargs):
        """
        :param max_length: maximum number word for 1 document
        :param overlapping: overlapping size for 2  document adjacency pair
        :param mode_chunking: on or off mode chunking long document
        :param corpus: dictionary context, question and answer
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
        list_documents = []
        for context in corpus:
            list_documents.extend(cls.get_documents(context, **kwargs))
        return cls(list_document=list_documents, **kwargs)

    @classmethod
    def chunk_document(cls, context: str, **kwargs):
        max_length = kwargs.get('max_length', cls.max_length)
        overlapping_size = kwargs.get('overlapping_size', cls.overlapping_size)
        size = max_length - overlapping_size
        list_words = context.split(" ")
        n_chunk = math.ceil(len(list_words) / size)
        list_context = []
        for i in range(n_chunk):
            temp_context = " ".join(list_words[i * size: i * size + max_length])
            list_context.append(temp_context)
        return list_context

    @classmethod
    def parser_uit_squad(cls, path_data: str, **kwargs):
        data = load_json_data(path_data)
        list_document = []
        if kwargs.get('mode_chunking', False):
            logger.info("Turn on mode chunkng long document")
        for context in data['data']:
            for paragraph in context['paragraphs']:
                list_document.extend(
                    cls.get_documents(paragraph, **kwargs)
                )

        return cls(list_document=list_document, **kwargs)

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
