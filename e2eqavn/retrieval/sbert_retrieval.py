from abc import ABC
from typing import *

import numpy as np
import math
from tqdm import tqdm

from e2eqavn.retrieval import BaseRetrieval
from e2eqavn.documents import *
from e2eqavn.utils.calculate import get_top_k_retrieval
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SentenceEvaluator
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.adamw import AdamW
import logging

logger = logging.getLogger(__name__)


class SentenceBertLearner:
    def __init__(self, model: SentenceTransformer, max_seq_length: int):
        self.model = model
        self.max_seq_length = max_seq_length

    @classmethod
    def from_pretrained(cls, model_name_or_path, max_seq_length: int = 512):
        try:
            model = SentenceTransformer(
                model_name_or_path=model_name_or_path,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        except:
            raise Exception(f"Can't load pretrained model sentence bert {model_name_or_path}")
        return cls(model, max_seq_length)

    def train(self, train_dataset: Dataset, loss_fn: nn.Module,
              dev_evaluator: Union[InformationRetrievalEvaluator, SentenceEvaluator] = None,
              batch_size: int = 16, epochs: int = 10, use_amp: bool = True,
              model_save_path: str = "Model", scheduler: str = 'WarmupLinear',
              warmup_steps: int = 1000, optimizer_class: Type[Optimizer] = AdamW,
              optimizer_params: Dict[str, object] = {'lr': 2e-5}, weight_decay: float = 0.01,
              max_grad_norm: float = 1, show_progress_bar: bool = True,
              save_best_model: bool = True, evaluation_steps: int = 5000
              ):
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size
        )
        self.model.fit(
            train_objectives=[(train_loader, loss_fn)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=model_save_path,
            evaluator=dev_evaluator,
            evaluation_steps=evaluation_steps,
            use_amp=use_amp,
            optimizer_params=optimizer_params,
            optimizer_class=optimizer_class,
            scheduler=scheduler,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            save_best_model=save_best_model,
            show_progress_bar=show_progress_bar
        )

        self.model.save(path=model_save_path)

    def encode_context(self, sentences: List[str], batch_size: int = 64,
                       show_progress_bar: bool = False, output_value: str = 'sentence_embedding',
                       convert_to_numpy: bool = False, convert_to_tensor: bool = False,
                       normalize_embeddings: bool = False, device: torch.device = None, **kwargs):
        if device is None:
            device = next(self.model.parameters()).device
        return self.model.encode(
            sentences=sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            output_value=output_value,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
            normalize_embeddings=normalize_embeddings,
            device=device
        )

    def get_device(self):
        return next(self.model.parameters()).device


class SBertRetrieval(BaseRetrieval, ABC):
    def __init__(self, model: SentenceBertLearner, device,
                 corpus: Corpus = None,
                 corpus_embedding: Union[np.array, torch.Tensor] = None,
                 convert_to_numpy: bool = False,
                 convert_to_tensor: bool = False):
        self.list_documents = None
        self.model = model
        self.device = device
        self.corpus = corpus
        self.corpus_embedding = corpus_embedding
        self.convert_to_tensor = convert_to_tensor
        self.convert_to_numpy = convert_to_numpy
        if not convert_to_numpy and not convert_to_tensor:
            self.convert_to_numpy = True
        elif next(self.model.get_device()) == torch.device('cuda'):
            self.convert_to_tensor = True

    def retrieval(self, queries: List[str], top_k: int, **kwargs) -> List[List[Document]]:
        if kwargs.get("documents", None):
            index_selection = [[doc.index for doc in list_doc] for list_doc in kwargs.get('documents')]
        else:
            index_selection = None
        if kwargs.get('top_k_sbert', None):
            top_k = kwargs.get('top_k_sbert')
        scores, top_k_indexs = self.query_by_embedding(queries, top_k=top_k,
                                                       index_selection=index_selection, **kwargs)
        scores = scores.cpu().numpy()
        top_k_indexs = top_k_indexs.cpu().numpy()
        result = []
        for i in range(len(queries)):
            tmp = []
            for j in range(top_k - 1, -1, -1):
                idx = top_k_indexs[i][j]
                self.list_documents[idx].embedding_similarity_score = scores[i][j]
                tmp.append(self.list_documents[idx])
            result.append(tmp)
        # for i, idx in enumerate(indexs_result):
        #     self.corpus.list_document[idx].embedding_similarity_score = similarity_score[i]
        #     result.append(self.corpus.list_document[idx])
        return result

    def update_embedding(self, corpus: Corpus, batch_size: int = 64, **kwargs):
        """
        Update embedding for corpus
        :param corpus: Corpus document context
        :param batch_size: number document in 1 batch
        :return:
        """
        logger.info(f"Start encoding corpus with {len(corpus.list_document)} document")
        n_docs = len(corpus.list_document)
        n_batch = math.ceil(n_docs / batch_size)
        corpus_embedding = []
        self.corpus = corpus
        self.list_documents = corpus.list_document
        for i in tqdm(range(n_batch)):
            sentences = [doc.document_context
                         for doc in corpus.list_document[
                                    batch_size * i: min(batch_size * (i + 1),
                                                        n_docs)]]
            embeddings = self.model.encode_context(
                sentences=sentences,
                batch_size=batch_size,
                convert_to_numpy=False,
                convert_to_tensor=True,
                device=self.device
            )
            corpus_embedding.append(embeddings)
        self.corpus_embedding = torch.concat(corpus_embedding, dim=0)

    def query_by_embedding(self, query: List[str], top_k: int, **kwargs):
        """
        :param top_k: k index document will return
        :param query: question
        :return: List document id
        """
        if kwargs.get('index_selection', None):
            index_selection = torch.tensor(kwargs.get('index_selection')).to(self.device)
        else:
            index_selection = None

        query_embedding = self.model.encode_context(
            sentences=query,
            convert_to_tensor=True,
            convert_to_numpy=False,
            device=self.device
        )

        similarity = util.cos_sim(query_embedding, self.corpus_embedding)
        if index_selection is not None:
            similarity = similarity.take(index_selection)
            scores, index = torch.topk(similarity, top_k, dim=1, sorted=True)
            sub_index_select = index_selection.take(index)
        else:
            scores, sub_index_select = torch.topk(similarity, top_k, dim=1, sorted=True)
        return scores, sub_index_select

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceBertLearner.from_pretrained(model_name_or_path)
        return cls(model=model, device=device)
