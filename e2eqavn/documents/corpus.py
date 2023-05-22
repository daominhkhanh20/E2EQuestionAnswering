from typing import List, Dict, Optional, Text, Union
import unicodedata
import numpy as np
import math
import logging
import hashlib
from torch import Tensor
from collections import defaultdict
from e2eqavn.utils.io import load_json_data, write_json_file
from e2eqavn.keywords import *
from e2eqavn.utils.preprocess import process_text

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
    def __init__(self, document_id: str, document_context: str, question: str, list_dict_answer: List[Dict]):
        self.document_id = document_id
        self.document_context = document_context
        self.question = question
        self.list_dict_answer = list_dict_answer

    # def find_index_answer(self):
    #     """
    #     Method find span answer index base on tokenizer for Machine Reading Comprehension task
    #     :return: answer_start and answer_end
    #     """
    #     raise NotImplementedError()


class Document:
    def __init__(self, document_context: str, document_id: str = None,
                 list_pair_question_answers: List[PairQuestionAnswers] = None,
                 embedding: Union[np.array, Tensor] = None,
                 index: int = 0,
                 bm25_score: float = 0,
                 embedding_similarity_score: float = 0,
                 score: float = 0,
                 pyvi_mode: bool = False):
        self.document_context = document_context
        self.index = index
        # if pyvi_mode:
        #     self.document_context = ViTokenizer.tokenize(self.document_context)
        self.bm25_score = bm25_score
        self.embedding_similarity_score = embedding_similarity_score
        self.score = score
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
            for question, list_dict_answer in dict_question_answers.items():
                temp.append(
                    PairQuestionAnswers(
                        document_id=document_id,
                        document_context=document_context,
                        question=question,
                        list_dict_answer=list_dict_answer
                    )
                )
            return cls(document_context=document_context, document_id=document_id, list_pair_question_answers=temp,
                       index=index)
        return cls(document_context=document_context, document_id=document_id, list_pair_question_answers=[],
                   index=index)


class Corpus:
    context_key: str = CONTEXT
    qas_key: str = QAS
    question_key: str = QUESTION
    answers_key: str = ANSWERS
    answer_key: str = TEXT
    max_length_document: int = 400
    overlapping_size: int = 40
    answer_start = ANSWER_START

    def __init__(self, list_document: List[Document], **kwargs):
        self.list_document = list_document
        self.n_document = len(self.list_document)
        self.n_pair_question_answer = 0
        self.list_document_context = [document.document_context for document in list_document]
        for document in self.list_document:
            self.n_pair_question_answer += len(document.list_pair_question_answers)
        self.__dict__.update(kwargs)

    @classmethod
    def get_documents(cls, context: Dict, doc_th: int = 0, **kwargs):
        context_key = kwargs.get(CONTEXT_KEY, cls.context_key)
        qas_key = kwargs.get(QAS_KEY, cls.qas_key)
        question_key = kwargs.get(QUESTION_KEY, cls.question_key)
        answers_key = kwargs.get(ANSWERS_KEY, cls.answers_key)
        answer_key = kwargs.get(ANSWER_KEY, cls.answer_key)
        infer_mode = kwargs.get(INFER_MODE, False)
        is_vnsquad_eval = kwargs.get(IS_VNSQUAD_EVAL, False)
        answer_start = kwargs.get(ANSWER_START, cls.answer_start)

        list_document = []
        document_context = context[context_key]
        if not kwargs.get(MODE_CHUNKING, False):
            document_id = hashlib.sha1(str(document_context).encode('utf-8')).hexdigest()
            dict_question_answers = defaultdict(list)
            if (not infer_mode) or is_vnsquad_eval:
                for question in context[qas_key]:
                    if not is_vnsquad_eval:
                        for answer in question[answers_key]:
                            dict_question_answers[question[question_key]].append(
                                {
                                    answer_key: answer[answer_key],
                                    answer_start: answer[answer_start]
                                }
                            )
                    else:
                        dict_question_answers[question[question_key]] = []
            list_document.append(
                Document.init_document(
                    document_id=document_id,
                    document_context=document_context,
                    dict_question_answers=dict_question_answers,
                    index=doc_th
                )
            )
            doc_th += 1
        else:
            list_context = cls.chunk_document(document_context, **kwargs)
            list_context_id = [hashlib.sha1(str(context).encode('utf-8')).hexdigest()
                               for context in list_context]
            dict_question_answers = {key: {} for key in list_context_id}
            over_lapping_size = kwargs.get(OVER_LAPPING_SIZE, cls.overlapping_size)
            if not infer_mode or is_vnsquad_eval:
                for question in context[qas_key]:
                    for answer in question[answers_key]:
                        flag_exist = False
                        n_char = 0
                        for idx, context_chunk in enumerate(list_context):
                            if answer[answer_key] in context_chunk:
                                if question[question_key] not in dict_question_answers[list_context_id[idx]]:
                                    dict_question_answers[list_context_id[idx]][question[question_key]] = [
                                        {
                                            answer_key: answer[answer_key],
                                            answer_start: answer[answer_start] - n_char
                                        }
                                    ]
                                else:
                                    dict_question_answers[list_context_id[idx]][question[question_key]].append(
                                        {
                                            answer_key: answer[answer_key],
                                            answer_start: answer[answer_start] - n_char
                                        }
                                    )
                                flag_exist = True
                                break
                            if idx == 0:
                                n_char += len(context_chunk)
                            else:
                                n_char += len(" ".join(context_chunk.split(" ")[over_lapping_size:]).strip())

                        if not flag_exist:
                            logger.info(f"Answer: {answer[answer_key]} \n "
                                        f"N chunk context: {len(list_context)}\n"
                                        f"List Context: {list_context} \n"
                                        f"Answer doesn't exist in context\n\n")
            else:
                for key in list_context_id:
                    dict_question_answers[key] = {}

            for idx, (key, value) in enumerate(dict_question_answers.items()):
                list_document.append(
                    Document.init_document(
                        document_id=key,
                        document_context=list_context[idx],
                        dict_question_answers=dict_question_answers[list_context_id[idx]],
                        index=doc_th
                    )
                )
                doc_th += 1
        return list_document, doc_th

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
        doc_th = 0
        for context in corpus:
            tmp_list_documents, doc_th = cls.get_documents(context, doc_th, **kwargs)
            list_documents.extend(tmp_list_documents)
        return cls(list_document=list_documents, **kwargs)

    @classmethod
    def chunk_document(cls, context: str, **kwargs):
        max_length = kwargs.get(MAX_LENGTH_DOCUMENT, cls.max_length_document)
        overlapping_size = kwargs.get(OVER_LAPPING_SIZE, cls.overlapping_size)
        size = max_length - overlapping_size
        list_words = context.split(" ")
        n_chunk = math.ceil(len(list_words) / size)
        list_context = []
        for i in range(n_chunk):
            temp_context = " ".join(list_words[i * size: i * size + max_length])
            list_context.append(temp_context)
            if len(temp_context.split(" ")) > max_length:
                print(max_length)
                print("MDLSAMKDMLASDMOASKSAkd")
        return list_context

    @classmethod
    def parser_uit_squad(cls, path_data: str, **kwargs):
        data = load_json_data(path_data)
        doc_th = 0
        list_document = []
        if kwargs.get('mode_chunking', False):
            logger.info("Turn on mode chunkng long document")
            logger.info(f"Max length for 1 document: {kwargs.get(MAX_LENGTH_DOCUMENT, cls.max_length_document)}")
        for context in data['data']:
            for paragraph in context['paragraphs']:
                tmp_list_documents, doc_th = cls.get_documents(paragraph, doc_th, **kwargs)
                list_document.extend(
                    tmp_list_documents
                )

            if len(list_document) > 20:
                break

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
