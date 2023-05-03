import os
import tempfile
import random
from transformers import AutoTokenizer
import logging

from datasets import load_dataset
from e2eqavn.documents import Corpus, Document
from e2eqavn.keywords import *
from e2eqavn.utils.calculate import calculate_input_training_for_qa
from e2eqavn.utils.preprocess import *
from e2eqavn.utils.io import write_json_file

logger = logging.getLogger(__name__)


class MRCDataset:
    def __init__(self, train_dataset, evaluator_dataset, **kwargs):
        self.train_dataset = train_dataset
        self.evaluator_dataset = evaluator_dataset

    @classmethod
    def make_dataset(cls, corpus: Corpus, mode: str, **kwargs):
        logger.info(f"Start prepare {mode} dataset")
        logger.info(f"Filter valid = {kwargs.get(IS_VALID, False)}")
        if MODEL_NAME_OR_PATH not in kwargs:
            raise Exception("You must provide pretrained name for QA")
        examples = []
        tokenizer = AutoTokenizer.from_pretrained(kwargs.get(MODEL_NAME_OR_PATH))
        is_document_right = kwargs.get(IS_DOCUMENT_RIGHT, True)
        num_proc = kwargs.get(NUM_PROC, 5)
        i = 0
        for document in corpus.list_document:
            if len(document.list_pair_question_answers) == 0:
                continue
            document_context = preprocess_qa_text(document.document_context)
            for question_answer in document.list_pair_question_answers:
                question = question_answer.question
                list_dict_answer = question_answer.list_dict_answer
                dict_answer = random.choice(list_dict_answer)
                answer = preprocess_qa_text(dict_answer['text'])
                examples.append(
                    {
                        CONTEXT: document_context,
                        ANSWER: preprocess_answer(document_context, answer, dict_answer[ANSWER_START]),
                        QUESTION: question,
                        ANSWER_START: dict_answer.get(ANSWER_START, None)
                    }
                )
        dir_save = kwargs.get(FOLDER_QA_SAVE, 'data/qa')
        if not os.path.exists(dir_save):
            os.makedirs(dir_save, exist_ok=True)
        examples = {'data': examples}
        write_json_file(examples, os.path.join(dir_save, f"{mode}.json"))
        dataset = load_dataset(
            'json',
            data_files={mode: os.path.join(dir_save, f"{mode}.json")},
            field='data'
        )
        dataset = dataset.shuffle().map(
            calculate_input_training_for_qa,
            batched=False,
            num_proc=num_proc,
            fn_kwargs={
                'tokenizer': tokenizer,
                'is_document_right': is_document_right
            }
        )
        if kwargs.get(IS_VALID, False):
            dataset = dataset.filter(lambda x: x[IS_VALID], num_proc=num_proc)

        return dataset[mode]

    @classmethod
    def init_mrc_dataset(cls, corpus_train: Corpus, corpus_eval: Corpus, **kwargs):
        train_dataset = cls.make_dataset(corpus_train, mode='train', **kwargs)
        if corpus_eval is not None:
            eval_dataset = cls.make_dataset(corpus_eval, mode='validation', **kwargs)
        else:
            eval_dataset = None
        return cls(train_dataset, eval_dataset)
