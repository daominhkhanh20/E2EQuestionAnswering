from e2eqavn.documents import Corpus
from e2eqavn.datasets import MRCDataset
from e2eqavn.utils.io import load_yaml_file
from e2eqavn.keywords import *

config = load_yaml_file('config/train_bm25.yaml')
config_qa = config['reader']
train_corpus = Corpus.parser_uit_squad(config_qa['data']['path_train'])
eval_corpus = Corpus.parser_uit_squad(config_qa['data']['path_evaluator'])
dataset = MRCDataset.init_mrc_dataset(
    corpus_train=train_corpus,
    corpus_eval=eval_corpus,
    model_name_or_path=config_qa['model'][MODEL_NAME_OR_PATH]
)

