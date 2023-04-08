import click
from typing import *
import os
import json
from e2eqavn import __version__
from e2eqavn.utils.io import load_yaml_file
from e2eqavn.documents import Corpus
from e2eqavn.processor import RetrievalGeneration


@click.group()
def entry_point():
    print(f"e2eqa version {__version__}")
    pass


@click.command()
@click.option(
    '--config', '-c',
    required=True,
    default='config/train.yaml',
    help='Path train'
)
def train(config: Union[str, Text]):
    config_pipeline = load_yaml_file(config)
    retrieval_config = config_pipeline['pipeline']['retrieval']
    reader_config = config_pipeline['pipeline']['reader']
    if retrieval_config['is_train']:
        corpus = Corpus.parser_uit_squad(**retrieval_config['data'])
        sampling = RetrievalGeneration.generate_sampling(corpus, **retrieval_config['data'])


entry_point.add_command(train)
