from e2eqavn.documents import Corpus
from e2eqavn.datasets import MRCDataset
from e2eqavn.utils.io import load_yaml_file
from e2eqavn.keywords import *
from e2eqavn.mrc import MRCReader

path_checkpoint = 'model-bin/model/qa/checkpoint-25432'
config = load_yaml_file('config/train_bm25.yaml')
config_qa = config['reader']
eval_corpus = Corpus.parser_uit_squad(config_qa['data']['path_evaluator'])
dataset = MRCDataset.init_mrc_dataset(
    corpus_train=None,
    corpus_eval=eval_corpus,
    model_name_or_path=config_qa['model'][MODEL_NAME_OR_PATH],
    id_valid=config_qa['parameters'].get('is_valid', False)
)
reader = MRCReader.from_pretrained(path_checkpoint)
reader.init_trainer(dataset, **config_qa['model'])
reader.evaluate(dataset.evaluator_dataset)
