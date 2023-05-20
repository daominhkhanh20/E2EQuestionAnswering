from e2eqavn.documents import Corpus
from e2eqavn.datasets import MRCDataset
from e2eqavn.utils.io import load_yaml_file, write_json_file
from e2eqavn.keywords import *
from e2eqavn.mrc import MRCReader
from e2eqavn.datasets import TripletDataset
from e2eqavn.processor import RetrievalGeneration
from e2eqavn.retrieval import SentenceBertLearner
from e2eqavn.utils.calculate import make_vnsquad_retrieval_evaluator
from e2eqavn.keywords import  *
import wandb
import os

mode = None
config_pipeline = load_yaml_file('config/train_qa.yaml')
train_corpus = Corpus.parser_uit_squad(
        config_pipeline[DATA][PATH_TRAIN],
        **config_pipeline.get(CONFIG_DATA, {})
    )
retrieval_config = config_pipeline.get(RETRIEVAL, None)
reader_config = config_pipeline.get(READER, None)
if (mode == 'retrieval' or mode is None) and retrieval_config:
    retrieval_sample = RetrievalGeneration.generate_sampling(train_corpus, **retrieval_config[PARAMETERS])
    train_dataset = TripletDataset.load_from_retrieval_sampling(retrieval_sample=retrieval_sample)
    dev_evaluator = make_vnsquad_retrieval_evaluator(
        path_data_json=config_pipeline[DATA][PATH_EVALUATOR]
    )

    retrieval_learner = SentenceBertLearner.from_pretrained(
        model_name_or_path=retrieval_config[MODEL].get(MODEL_NAME_OR_PATH, 'khanhbk20/vn-sentence-embedding')
    )
    retrieval_learner.train(
        train_dataset=train_dataset,
        dev_evaluator=dev_evaluator,
        **retrieval_config[MODEL]
    )

if (mode == 'reader' or mode is None) and reader_config:
    eval_corpus = Corpus.parser_uit_squad(
        config_pipeline[DATA][PATH_EVALUATOR],
        **config_pipeline.get(CONFIG_DATA, {})
    )
    mrc_dataset = MRCDataset.init_mrc_dataset(
        corpus_train=train_corpus,
        corpus_eval=eval_corpus,
        model_name_or_path=reader_config[MODEL].get(MODEL_NAME_OR_PATH, 'khanhbk20/mrc_testing'),
        max_length=reader_config[MODEL].get(MAX_LENGTH, 368)
    )
    reader_model = MRCReader.from_pretrained(
        model_name_or_path=reader_config[MODEL].get(MODEL_NAME_OR_PATH, 'khanhbk20/mrc_testing')
    )
    reader_model.init_trainer(mrc_dataset=mrc_dataset, **reader_config[MODEL])
    reader_model.train()