from nltk import word_tokenize
import logging
import re
import random
from e2eqavn.documents import Corpus
from e2eqavn.keywords import *

logger = logging.getLogger(__name__)


class QATextProcessor:
    def __init__(self, context_key: str = 'context',
                 question_key: str = 'question',
                 answer_key: str = 'answer',
                 answer_start_key: str = 'answer_start',
                 answer_word_start_idx_key: str = 'answer_word_start_idx',
                 answer_word_end_idx_key: str = 'answer_word_end_idx'):
        self.dict_word_map = {}
        self.context_key = context_key
        self.answer_key = answer_key
        self.question_key = question_key
        self.answer_start_key = answer_start_key
        self.answer_word_start_idx_key = answer_word_start_idx_key
        self.answer_word_end_idx_key = answer_word_end_idx_key
        self.cnt_failed = 0

    def string_tokenize(self, text: str):
        words = text.split(" ")
        list_words = []
        for word in words:
            if self.dict_word_map.get(word, None) is None:
                self.dict_word_map[word] = " ".join(word_tokenize(word)).replace('``', '"').replace("''", '"')
            list_words.append(self.dict_word_map[word])
        return list_words

    def strip_answer_string(self, text: str):
        text = text.strip()
        while text[-1] in '.,/><;:\'"[]{}+=-_)(*&^!~`':
            if text[0] != '(' and text[-1] == ')' and '(' in text:
                break
            if text[-1] == '"' and text[0] != '"' and text.count('"') > 1:
                break
            text = text[:-1].strip()
        while text[0] in '.,/><;:\'"[]{}+=-_)(*&^!~`':
            if text[0] == '"' and text[-1] != '"' and text.count('"') > 1:
                break
            text = text[1:].strip()
        text = text.strip()
        return text

    def strip_context(self, text: str):
        text = text.replace('\n', ' ')
        text = re.sub('\s+', ' ', text)
        return text.strip()

    def process_example(self, example: dict):
        question = example[self.question_key]
        context = example[self.context_key]
        answer = example[self.answer_key]
        answer_start_raw = example[self.answer_start_key]
        flag = False
        for step in [-1, 0, 1]:
            if context[answer_start_raw + step: answer_start_raw + step + len(answer)] == answer:
                answer_start_raw += step
                flag = True
                break
        if flag:
            context_previous = self.strip_context(context[: answer_start_raw])
            answer = self.strip_answer_string(answer)
            context_next = self.strip_context(context[answer_start_raw + len(answer):])

            context_previous = " ".join(self.string_tokenize(context_previous))
            context_next = " ".join(self.string_tokenize(context_next))
            answer = " ".join(self.string_tokenize(answer))
            question = " ".join(self.string_tokenize(question))

            context = f"{context_previous} {answer} {context_next}"
            answer_word_start_idx = len(context_previous.split(" "))
            answer_word_end_idx = answer_word_start_idx + len(answer.split(" ")) - 1
            example = {
                self.context_key: context,
                self.question_key: question,
                self.answer_key: answer,
                self.answer_word_start_idx_key: answer_word_start_idx,
                self.answer_word_end_idx_key: answer_word_end_idx,
                IS_VALID: True
            }
        else:
            logger.info(f"Answer isn't context\n"
                        f"Answer: {answer} \n"
                        f"Answer start: {answer_start_raw}\n"
                        f"Question: {question} \n"
                        f"Context: {context}\n")
            self.cnt_failed += 1
            context = " ".join(self.string_tokenize(context))
            example = {
                self.context_key: context,
                self.question_key: question,
                self.answer_key: "",
                self.answer_word_start_idx_key: 0,
                self.answer_word_end_idx_key: 0,
                IS_VALID: False
            }

        return example

    def make_example(self, corpus: Corpus):
        examples = []
        for document in corpus.list_document:
            if len(document.list_pair_question_answers) == 0:
                continue
            document_context = document.document_context
            for question_answer in document.list_pair_question_answers:
                question = question_answer.question
                list_dict_answer = question_answer.list_dict_answer
                dict_answer = random.choice(list_dict_answer)
                answer = dict_answer['text']
                example = self.process_example(
                    {
                        self.context_key: document_context,
                        self.question_key: question,
                        self.answer_key: answer,
                        self.answer_start_key: dict_answer.get(self.answer_start_key, None)
                    }
                )
                if not example[IS_VALID]:
                    continue
                examples.append(example)

        logger.info(f"*" * 50)
        logger.info(f"Total {self.cnt_failed} document failed")
        logger.info(f"*" * 50)
        return examples
