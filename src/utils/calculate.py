from typing import Union
import numpy as np
from numpy import array
from torch import Tensor
from sentence_transformers import util


def get_top_k_retrieval(query_embedding: Union[Tensor, array],
                        corpus_embedding: Union[Tensor, array],
                        top_k: int):
    similarity = util.cos_sim(query_embedding, corpus_embedding)
    if isinstance(similarity, Tensor):
        similarity = similarity.cpu().numpy()
    index_sorted = np.argsort(similarity, axis=1)[:, ::-1][:, :top_k]
    return index_sorted
