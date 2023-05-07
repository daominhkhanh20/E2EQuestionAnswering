from typing import *
from abc import ABC
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers import TrainingArguments, Trainer, AutoTokenizer

from .base import BaseReader
from e2eqavn.documents import Document
from e2eqavn.datasets import DataCollatorCustom, MRCDataset
from e2eqavn.keywords import *
from e2eqavn.evaluate import MRCEvaluator


class MRCQuestionAnsweringModel(RobertaPreTrainedModel, ABC):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = RobertaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(self, input_ids: Tensor, attention_mask: Tensor,
                start_positions: Tensor = None, end_positions: Tensor = None,
                return_dict: bool = None, start_idx: Tensor = None, end_idx: Tensor = None,
                words_length: Tensor = None, span_answer_ids: Tensor = None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids,
            attention_mask,
            return_dict=return_dict
        )
        sequence_output = outputs[0]
        batch_size = input_ids.size(0)
        n_sub_word = input_ids.size(1)
        n_word = words_length.size(1)
        align_matrix = torch.zeros(batch_size, n_word, n_sub_word)
        for i, sample_length in enumerate(words_length):
            for j in range(len(sample_length)):
                tmp_idx = torch.sum(sample_length[:j])
                align_matrix[i][j][tmp_idx: tmp_idx + sample_length[j]] = 1 if sample_length[j] > 0 else 0
        align_matrix = align_matrix.to(sequence_output.device)
        sequence_output = torch.bmm(align_matrix, sequence_output)
        outs = self.qa_outputs(sequence_output)
        start_logits, end_logits = outs.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        loss = None
        if start_positions is not None and end_positions is not None:
            ignore_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignore_index)
            end_positions = end_positions.clamp(0, ignore_index)
            loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
            loss = (
                           loss_fn(start_logits, start_positions) + loss_fn(end_logits, end_positions)
                   ) / 2
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )


class MRCReader(BaseReader, ABC):
    def __init__(self, model, tokenizer, device):
        self.compute_metrics = None
        self.trainer = None
        self.device = device
        self.model = model.to(device)
        self.tokenizer = tokenizer

    def encode(self, query: str, documents: List[Document], **kwargs):
        pass

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MRCQuestionAnsweringModel.from_pretrained(model_name_or_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained('khanhbk20/mrc_testing')
        return cls(model, tokenizer, device)

    def init_trainer(self, mrc_dataset: MRCDataset, **kwargs):
        training_args = TrainingArguments(
            output_dir=kwargs.get(OUTPUT_DIR, 'model/qa'),
            do_train=kwargs.get(DO_TRANING, True if mrc_dataset.train_dataset is not None else False),
            do_eval=kwargs.get(DO_EVAL, True if mrc_dataset.evaluator_dataset is not None else False),
            num_train_epochs=kwargs.get(NUM_TRAIN_EPOCHS, 20),
            learning_rate=float(kwargs.get(LEARNING_RATE, 1e-4)),
            warmup_ratio=kwargs.get(WARMPUP_RATIO, 0.05),
            weight_decay=kwargs.get(WEIGHT_DECAY, 0.01),
            per_device_train_batch_size=kwargs.get(BATCH_SIZE_TRAINING, 16),
            per_device_eval_batch_size=kwargs.get(BATCH_SIZE_EVAL, 32),
            gradient_accumulation_steps=kwargs.get(GRADIENT_ACCUMULATION_STEPS, 1),
            logging_dir='log',
            logging_strategy=kwargs.get(LOGGING_STRATEGY, 'epoch'),
            logging_steps=kwargs.get(LOGGING_STEP, 2),
            label_names=[
                "start_positions",
                "end_positions",
                "span_answer_ids",
                "input_ids",
                "words_length"
            ],
            group_by_length=True,
            save_strategy=kwargs.get(SAVE_STRATEGY, 'no'),
            metric_for_best_model=kwargs.get(METRIC_FOR_BEST_MODEL, 'f1'),
            load_best_model_at_end=kwargs.get(LOAD_BEST_MODEL_AT_END, True),
            save_total_limit=kwargs.get(SAVE_TOTAL_LIMIT, 2),
            evaluation_strategy=kwargs.get(EVALUATION_STRATEGY, 'epoch')
        )

        data_collator = DataCollatorCustom(tokenizer=self.tokenizer)
        self.compute_metrics = MRCEvaluator(tokenizer=self.tokenizer)
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=mrc_dataset.train_dataset,
            eval_dataset=mrc_dataset.evaluator_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )

    def train(self):
        self.trainer.train()

    def evaluate(self, dataset):
        self.trainer.evaluate(dataset)
        self.compute_metrics.save_log()

    def predict(self, query: Union[str, List[str]], documents: List[Document], **kwargs):
        pass
