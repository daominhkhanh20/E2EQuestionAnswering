from e2eqavn.documents import Corpus
from e2eqavn.datasets import MRCDataset
from e2eqavn.utils.io import load_yaml_file
from e2eqavn.keywords import *
from e2eqavn.mrc import MRCReader

config = load_yaml_file('config/train_bm25.yaml')
config_qa = config['reader']
train_corpus = Corpus.parser_uit_squad(config_qa['data']['path_train'])
eval_corpus = Corpus.parser_uit_squad(config_qa['data']['path_evaluator'])

# train_corpus = Corpus.parser_uit_squad(config_qa['data']['path_train'], **config_qa['parameters'])
# eval_corpus = Corpus.parser_uit_squad(config_qa['data']['path_evaluator'], **config_qa['parameters'])
dataset = MRCDataset.init_mrc_dataset(
    corpus_train=train_corpus,
    corpus_eval=eval_corpus,
    model_name_or_path=config_qa['model'][MODEL_NAME_OR_PATH],
    id_valid=config_qa['parameters'].get('is_valid', False)
)
print(len(dataset.train_dataset))
print(len(dataset.evaluator_dataset))
reader = MRCReader.from_pretrained(config_qa['model'][MODEL_NAME_OR_PATH])
reader.init_trainer(dataset, **config_qa['model'])
# reader.train()
loader = reader.trainer.get_train_dataloader()
sample = next(iter(loader))
outs = reader.model(**sample)
print(outs.keys())