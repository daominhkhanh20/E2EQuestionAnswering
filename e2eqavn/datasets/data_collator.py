from typing import *

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from e2eqavn.keywords import *


class DataCollatorCustom:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):

        def collate_fn(list_tensor: List[Tensor], padding_value: int):
            return pad_sequence(
                list_tensor,
                padding_value=padding_value,
                batch_first=True
            )

        input_ids = collate_fn(
            [
                torch.tensor(sample[INPUT_IDS]) for sample in batch
            ],
            padding_value=self.tokenizer.pad_token_id
        )
        attention_masks = collate_fn(
            [torch.tensor(sample[ATTENTION_MASK]) for sample in batch],
            padding_value=0
        )

        words_length = collate_fn(
            [torch.tensor(sample[WORDS_LENGTH]) for sample in batch],
            padding_value=0
        )
        if START_IDX in batch[0]:
            for sample in batch:
                start_idx = sum(sample[WORDS_LENGTH][: sample[START_IDX]])
                end_idx = sum(sample[WORDS_LENGTH][: sample[END_IDX] + 1])
                sample[SPAN_ANSWER_IDS] = sample[INPUT_IDS][start_idx: end_idx]
            span_answer_ids = collate_fn(
                [torch.tensor(sample[SPAN_ANSWER_IDS]) for sample in batch],
                padding_value=-100
            )
            start_idxs = torch.tensor([sample[START_IDX] for sample in batch])
            end_idxs = torch.tensor([sample[END_IDX] for sample in batch])
            return {
                'input_ids': input_ids,
                'attention_mask': attention_masks,
                'start_positions': start_idxs,
                'end_positions': end_idxs,
                'words_length': words_length,
                'span_answer_ids': span_answer_ids
            }

        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'words_length': words_length,
        }

# def data_collator_fn(samples):
#     def collate_fn(list_tensor: List[Tensor], padding_value: int):
#         return pad_sequence(
#             list_tensor,
#             padding_value=padding_value,
#             batch_first=True
#         )
#
#     input_ids = collate_fn(
#         [
#             torch.tensor(sample[INPUT_IDS]) for sample in samples
#         ],
#         padding_value=tokenizer.pad_token_id
#     )
#     attention_masks = collate_fn(
#         [torch.tensor(sample[ATTENTION_MASK]) for sample in samples],
#         padding_value=0
#     )
#     start_idxs = torch.tensor([sample['start_idx'] for sample in samples])
#     end_idxs = torch.tensor([sample['end_idx'] for sample in samples])
#     return {
#         'input_ids': input_ids,
#         'attention_mask': attention_masks,
#         'start_positions': start_idxs,
#         'end_positions': end_idxs
#     }
