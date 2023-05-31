from sentence_transformers import SentenceTransformer, util
from e2eqavn.documents import Corpus
from e2eqavn.datasets import MRCDataset
from e2eqavn.utils.io import load_yaml_file, write_json_file
from e2eqavn.keywords import *
from e2eqavn.mrc import MRCReader
import wandb
import os
import torch
from torch import nn

config_pipeline = load_yaml_file('config/train_qa.yaml')
train_corpus = Corpus.parser_uit_squad(
    config_pipeline[DATA][PATH_TRAIN],
    **config_pipeline.get(CONFIG_DATA, {})
)
model = SentenceTransformer('khanhbk20/vn-sentence-embedding')
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def make_input_sbert(sentence: str):
    return model.tokenize([sentence])


class SbertTritonModel(nn.Module):
    def __init__(self, corpus: Corpus):
        super().__init__()
        self.model = SentenceTransformer('khanhbk20/vn-sentence-embedding')
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.corpus_embedding = self.model.encode(
            sentences=[doc.document_context for doc in corpus.list_document],
            convert_to_tensor=True,
            convert_to_numpy=False,
            show_progress_bar=True,
            batch_size=64,
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
        print(sbert_index_selection)
        return sbert_index_selection, sbert_input_ids


sentence = 'Paris nằm ở điểm gặp nhau của các hành trình'
sbert_model = SbertTritonModel(corpus=train_corpus).eval()
input_feature = make_input_sbert(sentence)
torch.tensor([1, 2, 3, 4]).to(device)
traced_script_module = torch.jit.trace(sbert_model, (
    input_feature['input_ids'].to(device),
    input_feature['attention_mask'].to(device),
    input_feature['token_type_ids'].to(device),
    torch.tensor([[0, 2516, 339, 25, 192, 243, 138, 7, 9, 2537, 1893, 2]]).to(device),
    torch.tensor([2]).to(device)
)
                                       )
traced_script_module.save('model_compile/sbert/model.pt')
