from typing import *
from sentence_transformers import SentenceTransformer, util
from e2eqavn.documents import Corpus
from e2eqavn.datasets import MRCDataset
from e2eqavn.utils.io import load_yaml_file, write_json_file
from e2eqavn.keywords import *
from e2eqavn.mrc import MRCReader
import wandb
import json
import os
import torch
from pymongo import MongoClient
from torch import nn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--from_mongo', default=False, type=lambda x: x.lower() == 'true')

args = parser.parse_args()

if not args.from_mongo:
    config_pipeline = load_yaml_file('config/train_qa.yaml')
    corpus = Corpus.parser_uit_squad(
        config_pipeline[DATA][PATH_TRAIN],
        **config_pipeline.get(CONFIG_DATA, {})
    )
else:
    print('Load data from mongodb')
    client = MongoClient(
        "mongodb+srv://dataintergration:nhom10@cluster0.hqw7c.mongodb.net/test")
    database = client['wikipedia']
    MAX_LENGTH = 350
    OVERLAPPING_SIZE = 50
    wiki_collections_process = database[f'DocumentsProcess_{MAX_LENGTH}_{OVERLAPPING_SIZE}']
    corpus = []
    list_docs = []
    for document in wiki_collections_process.find():
        list_docs.append({
            'context': document['text'],
            'qas': []
        })
        corpus.append(document['text'])
        if len(list_docs) > 60000:
            break
    with open('model_compile/corpus.json', 'w') as file:
        json.dump(list_docs, file, indent=4, ensure_ascii=False)
    print("Load done")

model = SentenceTransformer('khanhbk20/vn-sentence-embedding')
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def make_input_sbert(sentence: str):
    return model.tokenize([sentence])


class SbertTritonModel(nn.Module):
    def __init__(self, corpus: Union[Corpus, List[str]]):
        super().__init__()
        self.model = SentenceTransformer('khanhbk20/vn-sentence-embedding')
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        if isinstance(corpus, Corpus):
            list_texts = [doc.document_context for doc in corpus.list_document]
        else:
            list_texts = corpus
        self.corpus_embedding = self.model.encode(
            sentences=list_texts,
            convert_to_tensor=True,
            convert_to_numpy=False,
            show_progress_bar=True,
            batch_size=128,
            device=self.device
        )

    def forward(self, sbert_input_ids, sbert_attention_mask, sbert_token_type_ids, bm25_index_selection, top_k_sbert):
        input_feature = {'input_ids': sbert_input_ids, 'attention_mask': sbert_attention_mask,
                         'token_type_ids': sbert_token_type_ids}
        embedding = self.model.forward(input_feature)['sentence_embedding']
        sub_corpus_embedding = self.corpus_embedding[bm25_index_selection.reshape(-1), :]
        sim_score = util.cos_sim(embedding, sub_corpus_embedding)
        scores, indexs = torch.topk(sim_score, top_k_sbert.item(), dim=1, largest=True, sorted=True)
        sbert_index_selection = bm25_index_selection[torch.arange(indexs.size(0)), indexs]
        return sbert_index_selection, sbert_input_ids


sentence = 'Cơ sở giáo dục phương Tây đầu tiên có'
sbert_model = SbertTritonModel(corpus=corpus).to(device).eval()
input_feature = make_input_sbert(sentence)
torch.tensor([1, 2, 3, 4]).to(device)
traced_script_module = torch.jit.trace(sbert_model, (
    input_feature['input_ids'].to(device),
    input_feature['attention_mask'].to(device),
    input_feature['token_type_ids'].to(device),
    torch.tensor([[0, 1, 339, 25, 192, 243, 138, 7, 9, 2537, 1893, 2]]).to(device),
    torch.tensor([2]).to(device)
)
                                       )
traced_script_module.save('model_compile/sbert/model.pt')
