import click
from typing import *
import os
import json
from e2eqavn import __version__
from e2eqavn.utils.io import load_yaml_file
from e2eqavn.documents import Corpus
from e2eqavn.datasets import *
from e2eqavn.processor import RetrievalGeneration
from e2eqavn.keywords import *
from e2eqavn.utils.calculate import *
from e2eqavn.retrieval import *
from e2eqavn.mrc import *
from e2eqavn.evaluate import *
from e2eqavn.utils.calculate import make_input_for_retrieval_evaluator
from e2eqavn.pipeline import E2EQuestionAnsweringPipeline


@click.group()
def entry_point():
    print(f"e2eqa version {__version__}")
    pass


@click.command()
def version():
    print(__version__)


@click.command()
@click.option(
    '--config', '-c',
    required=True,
    default='config/config.yaml',
    help='Path config model'
)
@click.argument('mode', default=None)
def train(config: Union[str, Text], mode: str):
    config_pipeline = load_yaml_file(config)
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


@click.command()
@click.option(
    '--config', '-c',
    required=True,
    default='config/config.yaml',
    help='Path config model'
)
@click.argument('mode', default='retrieval')
def evaluate(config: Union[str, Text], mode):
    config_pipeline = load_yaml_file(config)
    retrieval_config = config_pipeline.get(RETRIEVAL, None)
    reader_config = config_pipeline.get(READER, None)
    pipeline = E2EQuestionAnsweringPipeline()
    eval_corpus = Corpus.parser_uit_squad(
        config_pipeline[DATA][PATH_EVALUATOR],
        **config_pipeline.get(CONFIG_DATA, {})
    )
    if (mode == 'retrieval' or mode is None) and retrieval_config:
        corpus, queries, relevant_docs = make_input_for_retrieval_evaluator(
            path_data_json=config_pipeline[DATA][PATH_EVALUATOR]
        )
        retrieval_model = SBertRetrieval.from_pretrained(retrieval_config[MODEL][MODEL_NAME_OR_PATH])
        retrieval_model.update_embedding(eval_corpus)
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


@click.command()
@click.option(
    '--config', '-c',
    required=True,
    help='Path config model'
)
@click.option(
    '--question', '-q',
    required=True,
)
@click.option(
    '--top_k_bm25', '-bm25',
    default=10,
    help='Top k retrieval by BM25 algorithm'
)
@click.option(
    '--top_k_sbert', '-sbert',
    default=3,
    help='Top k retrieval by sentence-bert algorithm'
)
@click.argument('mode', default='retrieval')
def test(config: Union[str, Text], question: str, top_k_bm25: int, top_k_sbert: int, mode: str):
    config_pipeline = load_yaml_file(config)
    retrieval_config = config_pipeline.get(RETRIEVAL, None)
    reader_config = config_pipeline.get(READER, None)
    pipeline = E2EQuestionAnsweringPipeline()
    corpus = Corpus.parser_uit_squad(
        config_pipeline[DATA][PATH_TRAIN],
        **config_pipeline.get(CONFIG_DATA, {})
    )
    if (mode == 'retrieval' or mode is None) and retrieval_config:
        bm25_retrieval = BM25Retrieval(corpus=corpus)
        pipeline.add_component(
            component=bm25_retrieval,
            name_component='retrieval_1'
        )
        retrieval_model = SBertRetrieval.from_pretrained(retrieval_config[MODEL][MODEL_NAME_OR_PATH])
        retrieval_model.update_embedding(corpus=corpus)
        pipeline.add_component(
            component=retrieval_model,
            name_component='retrieval_2'
        )

    if (mode == 'reader' or mode is None) and reader_config:
        reader_model = MRCReader.from_pretrained(
            model_name_or_path=reader_config[MODEL].get(MODEL_NAME_OR_PATH, 'khanhbk20/mrc_testing')
        )
        pipeline.add_component(
            component=reader_model,
            name_component='reader'
        )
    return pipeline.run(
        queries=question,
        top_k_bm25=top_k_bm25,
        top_k_sbert=top_k_sbert
    )


entry_point.add_command(version)
entry_point.add_command(train)
entry_point.add_command(evaluate)
