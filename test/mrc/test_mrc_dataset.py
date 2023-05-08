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
print(train_corpus.list_document[0].document_context)
# train_corpus = Corpus.parser_uit_squad(config_qa['data']['path_train'], **config_qa['parameters'])
# eval_corpus = Corpus.parser_uit_squad(config_qa['data']['path_evaluator'], **config_qa['parameters'])
dataset = MRCDataset.init_mrc_dataset(
    corpus_train=train_corpus,
    corpus_eval=eval_corpus,
    model_name_or_path=config_qa['model'][MODEL_NAME_OR_PATH],
    id_valid=config_qa['parameters'].get('is_valid', False)
)
# print(len(dataset.train_dataset))
# print(len(dataset.evaluator_dataset))
# tokenizer = AutoTokenizer.from_pretrained(config_qa['model'][MODEL_NAME_OR_PATH])
# data_collator = DataCollatorCustom(tokenizer=tokenizer)
# print(dataset.train_dataset.column_names)

# # for sample in dataset.train_dataset:
# #     if len(sample['input_ids']) == 243:
# #         print(sample['context'])
# #         print(sample['question'])
# #         print(sample['answer'])
# loader = DataLoader(
#     dataset=dataset.train_dataset,
#     collate_fn=data_collator,
#     batch_size=16
# )
# for sample in dataset.train_dataset:
#     if len(sample['input_ids']) > 512:
#         print(len(sample['input_ids']))
# print('*'*50)
# for idx, sample in enumerate(loader):
#     if sample['input_ids'].size(1) > 512:
#         print(sample['input_ids'].size(1))
# # sample = next(iter(loader))
# for key, value in sample.items():
#     print(f"{key} {value.size()}")



