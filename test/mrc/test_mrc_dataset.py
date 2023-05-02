from e2eqavn.documents import Corpus
from e2eqavn.datasets import MRCDataset
from e2eqavn.utils.io import load_yaml_file
from e2eqavn.keywords import *
from e2eqavn.datasets import DataCollatorCustom
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch

config = load_yaml_file('config/train_bm25.yaml')
config_qa = config['reader']
train_corpus = Corpus.parser_uit_squad(config_qa['data']['path_train'])
eval_corpus = Corpus.parser_uit_squad(config_qa['data']['path_evaluator'])
dataset = MRCDataset.init_mrc_dataset(
    corpus_train=train_corpus,
    corpus_eval=eval_corpus,
    model_name_or_path=config_qa['model'][MODEL_NAME_OR_PATH]
)
tokenizer = AutoTokenizer.from_pretrained(config_qa['model'][MODEL_NAME_OR_PATH])
data_collator = DataCollatorCustom(tokenizer=tokenizer)
print(isinstance(dataset.train_dataset, torch.utils.data.IterableDataset))
print(dataset.train_dataset.column_names)
for sample in dataset.train_dataset:
    if len(sample['input_ids']) == 243:
        print(sample['context'])
        print(sample['question'])
        print(sample['answer'])
# loader = DataLoader(
#     dataset=dataset.train_dataset,
#     collate_fn=data_collator,
#     batch_size=3
# )
# print(next(iter(loader)).keys())
# print(dataset.train_dataset[0])



