from e2eqavn.documents import Corpus
from e2eqavn.datasets import MRCDataset
from e2eqavn.utils.io import load_yaml_file, write_json_file
from e2eqavn.keywords import *
from e2eqavn.mrc import MRCReader
import wandb
import os


config = load_yaml_file('config/train_qa.yaml')
config_qa = config['reader']
# train_corpus = Corpus.parser_uit_squad(config_qa['data']['path_train'])
# eval_corpus = Corpus.parser_uit_squad(config_qa['data']['path_evaluator'])
if not os.path.exists(config_qa['model'].get(OUTPUT_DIR, 'model/qa')):
    os.makedirs(config_qa['model'].get(OUTPUT_DIR, 'model/qa'))
write_json_file(config_qa, os.path.join(config_qa['model'].get(OUTPUT_DIR, 'model/qa'), 'parameter.json'))

config = {**config_qa['parameters'], **config_qa['model']}
wandb.init(project='E2E_QA_THESIS', config=config)
train_corpus = Corpus.parser_uit_squad(config_qa['data']['path_train'], **config_qa['parameters'])
eval_corpus = Corpus.parser_uit_squad(config_qa['data']['path_evaluator'], **config_qa['parameters'])
dataset = MRCDataset.init_mrc_dataset(
    corpus_train=train_corpus,
    corpus_eval=eval_corpus,
    model_name_or_path=config_qa['model'][MODEL_NAME_OR_PATH],
    max_length=config_qa['parameters'].get('max_length', 400),
    mode_chunking=config_qa['parameters'].get('mode_chunking', False)
)
# print(len(dataset.train_dataset))
# print(len(dataset.evaluator_dataset))
reader = MRCReader.from_pretrained(config_qa['model'][MODEL_NAME_OR_PATH])
reader.init_trainer(dataset, **config_qa['model'])
reader.train()
# loader = reader.trainer.get_train_dataloader()
# sample = next(iter(loader))
# outs = reader.model(**sample)
# print(outs.keys())