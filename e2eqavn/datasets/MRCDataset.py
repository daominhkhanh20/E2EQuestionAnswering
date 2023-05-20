import os
import tempfile
import random
from transformers import AutoTokenizer
import logging
import wandb
from datasets import load_dataset
from e2eqavn.documents import Corpus, Document
from e2eqavn.keywords import *
from e2eqavn.utils.calculate import *
from e2eqavn.utils.preprocess import *
from e2eqavn.utils.io import write_json_file
from e2eqavn.processor import QATextProcessor

logger = logging.getLogger(__name__)


class MRCDataset:
    def __init__(self, train_dataset, evaluator_dataset, **kwargs):
        self.train_dataset = train_dataset
        self.evaluator_dataset = evaluator_dataset

    @classmethod
    def make_dataset(cls, corpus: Corpus, mode: str, **kwargs):
        logger.info(f"Start prepare {mode} dataset")
        logger.info(f"Max length sentence = {kwargs.get(MAX_LENGTH, 512)}")
        if MODEL_NAME_OR_PATH not in kwargs:
            raise Exception("You must provide pretrained name for QA")
        tokenizer = AutoTokenizer.from_pretrained(kwargs.get(MODEL_NAME_OR_PATH))
        num_proc = kwargs.get(NUM_PROC, 5)
        qa_text_processor = QATextProcessor(
            context_key=kwargs.get(CONTEXT_KEY, 'context'),
            question_key=kwargs.get(QUESTION_KEY, 'question'),
            answer_key=kwargs.get(ANSWER_KEY, 'answer'),
            answer_start_key=kwargs.get(ANSWER_START, 'answer_start'),
            answer_word_start_idx_key=kwargs.get(ANSWER_WORD_START_IDX, 'answer_word_start_idx'),
            answer_word_end_idx_key=kwargs.get(ANSWER_WORD_END_IDX, 'answer_word_end_idx')
        )
        examples = qa_text_processor.make_example(corpus)
        dir_save = kwargs.get(FOLDER_QA_SAVE, 'data/qa')
        if not os.path.exists(dir_save):
            os.makedirs(dir_save, exist_ok=True)
        logger.info(f"Dataset for {mode} has {len(examples)} sample")
        examples = {'data': examples}
        write_json_file(examples, os.path.join(dir_save, f"{mode}.json"))
        dataset = load_dataset(
            'json',
            data_files={mode: os.path.join(dir_save, f"{mode}.json")},
            field='data'
        )

        dataset = dataset.shuffle().map(
            calculate_input_training_for_qav2,
            batched=False,
            num_proc=num_proc,
            fn_kwargs={
                'tokenizer': tokenizer,
                'max_length': kwargs.get(MAX_LENGTH, 368)
            }
        ).filter(lambda x: x['is_valid'], num_proc=num_proc)

        return dataset[mode]

    @classmethod
    def init_mrc_dataset(cls, corpus_train: Corpus = None, corpus_eval: Corpus = None, **kwargs):
        if corpus_train is not None:
            train_dataset = cls.make_dataset(corpus_train, mode='train', **kwargs)
        else:
            train_dataset = None

        if corpus_eval is not None:
            eval_dataset = cls.make_dataset(corpus_eval, mode='validation', **kwargs)
        else:
            eval_dataset = None
        return cls(train_dataset, eval_dataset)
