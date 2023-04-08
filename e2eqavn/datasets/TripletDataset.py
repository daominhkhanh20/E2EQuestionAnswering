from typing import *
from e2eqavn.processor import RetrievalGeneration

import torch
from torch.utils.data import IterableDataset
from sentence_transformers import InputExample


class TripletDataset(IterableDataset):
    def __init__(self, triplet_examples: List[Dict],
                 question_key: str = 'question',
                 positive_key: str = 'pos',
                 negative_key: str = 'neg'):
        self.triplet_example = triplet_examples
        self.question_key = question_key
        self.positive_key = positive_key
        self.negative_key = negative_key

    def __len__(self):
        return len(self.triplet_example)

    def __iter__(self):
        for example in self.triplet_example:
            yield InputExample(texts=[
                example[self.question_key],
                example[self.positive_key],
                example[self.negative_key]
            ])

    @classmethod
    def load_from_retrieval_sampling(cls, retrieval_sample: RetrievalGeneration):
        triplet_example = []
        for sample in retrieval_sample.list_retrieval_sample:
            triplet_example.append({
                "question": sample.question,
                "pos": sample.document_positive,
                "neg": sample.document_negative
            })
        return cls(triplet_example, question_key="question", positive_key="pos", negative_key="neg")
