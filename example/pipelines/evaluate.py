from e2eqavn.documents import Corpus
from e2eqavn.datasets import MRCDataset
from e2eqavn.utils.io import load_yaml_file, write_json_file
from e2eqavn.keywords import *
from e2eqavn.mrc import MRCReader
from e2eqavn.evaluate import *
from e2eqavn.utils.calculate import make_input_for_retrieval_evaluator
from e2eqavn.pipeline import E2EQuestionAnsweringPipeline
from e2eqavn.utils.calculate import make_input_for_retrieval_evaluator
from e2eqavn.keywords import *
import wandb
import os

mode = 'retrieval'
config_pipeline = load_yaml_file('config/train_qa.yaml')
retrieval_config = config_pipeline.get(RETRIEVAL, None)
reader_config = config_pipeline.get(READER, None)
pipeline = E2EQuestionAnsweringPipeline()
if (mode == 'retrieval' or mode is None) and retrieval_config:
    corpus, queries, relevant_docs = make_input_for_retrieval_evaluator(
        path_data_json=config_pipeline[DATA][PATH_EVALUATOR]
    )
    retrieval_model = SBertRetrieval.from_pretrained(retrieval_config[MODEL][MODEL_NAME_OR_PATH])
    pipeline.add_component(
        component=retrieval_model,
        name_component='retrieval'
    )
    information_evaluator = InformationRetrievalEvaluatorCustom(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs
    )
    information_evaluator.compute_metrices_retrieval(
        pipeline=pipeline
    )

if (mode == 'reader' or mode is None) and reader_config:
    eval_corpus = Corpus.parser_uit_squad(
        config_pipeline[DATA][PATH_EVALUATOR],
        **config_pipeline.get(CONFIG_DATA, {})
    )
    mrc_dataset = MRCDataset.init_mrc_dataset(
        corpus_eval=eval_corpus,
        model_name_or_path=reader_config[MODEL].get(MODEL_NAME_OR_PATH, 'khanhbk20/mrc_testing'),
        max_length=reader_config[MODEL].get(MAX_LENGTH, 368)
    )
    reader_model = MRCReader.from_pretrained(
        model_name_or_path=reader_config[MODEL].get(MODEL_NAME_OR_PATH, 'khanhbk20/mrc_testing')
    )
    reader_model.init_trainer(mrc_dataset=mrc_dataset, **reader_config[MODEL])
    reader_model.evaluate(mrc_dataset.evaluator_dataset)
