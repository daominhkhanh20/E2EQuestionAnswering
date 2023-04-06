from abc import ABC
from typing import *

from src.retrieval import BaseRetrieval
from src.documents import *
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers.losses import MultipleNegativesRankingLoss
from torch.optim.optimizer import Optimizer
from torch.optim.adamw import AdamW


class SentenceBertModel:
    def __init__(self, model: SentenceTransformer, max_seq_length: int):
        self.model = model
        self.max_seq_length = max_seq_length

    @classmethod
    def from_pretrained(cls, model_name_or_path, max_seq_length: int = 512):
        try:
            model = SentenceTransformer(model_name_or_path=model_name_or_path)
        except:
            raise Exception(f"Can't load pretrained model sentence bert {model_name_or_path}")
        return cls(model, max_seq_length)

    def train_contrastive_loss(self, train_dataset: Dataset, dev_evaluator=None,
                               batch_size: int = 16, epochs: int = 10, use_amp: bool = True,
                               model_save_path: str = "Model", scheduler: str = 'WarmupLinear',
                               warmup_steps: int = 1000, optimizer_class: Type[Optimizer] = AdamW,
                               optimizer_params: Dict[str, object] = {'lr': 2e-5}, weight_decay: float = 0.01,
                               max_grad_norm: float = 1, show_progress_bar: bool = True,
                               save_best_model: bool = True, evaluation_steps: int = 5000
                               ):
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        loss_fn = MultipleNegativesRankingLoss(model=self.model)
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
                       normalize_embeddings: bool = False, device: torch.device = None):
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


class SBertRetrieval(BaseRetrieval, ABC):
    def __init__(self, model: SentenceBertModel):
        self.model = model

    def retrieval(self, query: str, top_k: int, **kwargs) -> List[Document]:
        pass

    def update_embedding(self, corpus: Corpus, batch_size: int = 64):
        """
        Update embedding for corpus
        :param corpus: Corpus document context
        :param batch_size: number document in 1 batch
        :return:
        """
        pass

    def query_by_embedding(self, query: List[str]):
        """
        :param query: question
        :return: List document id
        """
        pass

    @classmethod
    def from_pretrained(cls, model_name_or_path: str):
        model = SentenceBertModel.from_pretrained(model_name_or_path)
        return cls(model=model)

