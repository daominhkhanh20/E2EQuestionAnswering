from typing import *

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from e2eqavn.keywords import *


class DataCollatorCustom:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, samples, **kwargs):
        def collate_fn(list_tensor: List[Tensor], padding_value: int):
            return pad_sequence(
                list_tensor,
                padding_value=padding_value,
                batch_first=True
            )

        input_ids = collate_fn(
            [
                torch.tensor(sample[INPUT_IDS]) for sample in samples
            ],
            padding_value=self.tokenizer.pad_token_id
        )
        attention_masks = collate_fn(
            [torch.tensor(sample[ATTENTION_MASK]) for sample in samples],
            padding_value=0
        )
        start_idxs = torch.tensor([sample[START_IDX] for sample in samples])
        end_idxs = torch.tensor([sample[END_IDX] for sample in samples])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'start_positions': start_idxs,
            'end_positions': end_idxs
        }

