from e2eqavn.documents import Corpus
from e2eqavn.datasets import MRCDataset
from e2eqavn.utils.io import load_yaml_file
from e2eqavn.keywords import *
from e2eqavn.mrc import MRCReader

config = load_yaml_file('config/train_bm25.yaml')
config_qa = config['reader']
train_corpus = Corpus.parser_uit_squad(config_qa['data']['path_train'])
eval_corpus = Corpus.parser_uit_squad(config_qa['data']['path_evaluator'])
dataset = MRCDataset.init_mrc_dataset(
    corpus_train=train_corpus,
    corpus_eval=eval_corpus,
    model_name_or_path=config_qa['model'][MODEL_NAME_OR_PATH]
)
print(len(dataset.train_dataset))
print(len(dataset.evaluator_dataset))
reader = MRCReader.from_pretrained(config_qa['model'][MODEL_NAME_OR_PATH])
reader.train(dataset)