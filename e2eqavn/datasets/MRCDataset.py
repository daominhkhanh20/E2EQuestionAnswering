import os
import tempfile
import random
from transformers import AutoTokenizer
import logging

from datasets import load_dataset, Dataset
from e2eqavn.documents import Corpus, Document
from e2eqavn.keywords import *
from e2eqavn.utils.calculate import calculate_input_training_for_qa
from e2eqavn.utils.io import write_json_file

logger = logging.getLogger(__name__)


class MRCDataset:
    def __init__(self, train_dataset: Dataset, evaluator_dataset: Dataset, **kwargs):
        self.train_dataset = train_dataset
        self.evaluator_dataset = evaluator_dataset

    @classmethod
    def make_dataset(cls, corpus: Corpus, mode: str, **kwargs):
        logger.info(f"Start prepare {mode} dataset")
        if MODEL_NAME_OR_PATH not in kwargs:
            raise Exception("You must provide pretrained name for QA")
        examples = []
        tokenizer = AutoTokenizer.from_pretrained(kwargs.get(MODEL_NAME_OR_PATH))
        is_document_right = kwargs.get(IS_DOCUMENT_RIGHT, True)
        num_proc = kwargs.get(NUM_PROC, 10)
        i = 0
        for document in corpus.list_document:
            if len(document.list_pair_question_answers) == 0:
                continue
            document_context = document.document_context
            for question_answer in document.list_pair_question_answers:
                question = question_answer.question
                list_answers = question_answer.list_answers
                answer = random.choice(list_answers)
                examples.append(
                    {
                        CONTEXT: document_context,
                        ANSWER: answer,
                        QUESTION: question,
                    }
                )
                i += 1
            if i >= 10:
                break
        dir_save = kwargs.get(FOLDER_QA_SAVE, 'data/qa')
        if not os.path.exists(dir_save):
            os.makedirs(dir_save, exist_ok=True)
        write_json_file(examples, os.path.join(dir_save, f"{mode}.json"))
        dataset = load_dataset(
            'json',
            data_files=os.path.join(dir_save, f"{mode}.json")
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

        return dataset

    @classmethod
    def init_mrc_dataset(cls, corpus_train: Corpus, corpus_eval: Corpus, **kwargs):
        train_dataset = cls.make_dataset(corpus_train, mode='train', **kwargs)
        if corpus_eval is not None:
            eval_dataset = cls.make_dataset(corpus_eval, mode='val', **kwargs)
        else:
            eval_dataset = None
        return cls(train_dataset, eval_dataset)
