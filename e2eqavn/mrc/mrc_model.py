from typing import *
from abc import ABC
import torch
import os
from torch import nn, Tensor
from transformers import RobertaPreTrainedModel, RobertaModel, RobertaConfig
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers import TrainingArguments, Trainer, AutoTokenizer
from e2eqavn.mrc import BaseReader
from e2eqavn.documents import Document
from e2eqavn.utils.calculate import *
from e2eqavn.utils.io import load_json_data, write_json_file
from e2eqavn.datasets import DataCollatorCustom, MRCDataset
from e2eqavn.keywords import *
from e2eqavn.evaluate import MRCEvaluator
import wandb


class MRCQuestionAnsweringModel(RobertaPreTrainedModel, ABC):
    config_class = RobertaConfig

    # _keys_to_ignore_on_load_unexpected = [r"pooler"]
    # _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self,
                input_ids: Tensor = None,
                words_length: Tensor = None,
                start_idx=None,
                end_idx=None,
                attention_mask: Tensor = None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                start_positions: Tensor = None,
                end_positions: Tensor = None,
                span_answer_ids=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None, ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        context_embedding = sequence_output

        # Compute align word sub_word matrix
        batch_size = input_ids.shape[0]
        max_sub_word = input_ids.shape[1]
        max_word = words_length.shape[1]
        align_matrix = torch.zeros((batch_size, max_word, max_sub_word))

        for i, sample_length in enumerate(words_length):
            for j in range(len(sample_length)):
                start_idx = torch.sum(sample_length[:j])
                align_matrix[i][j][start_idx: start_idx + sample_length[j]] = 1 if sample_length[j] > 0 else 0

        align_matrix = align_matrix.to(context_embedding.device)
        # Combine sub_word features to make word feature
        context_embedding_align = torch.bmm(align_matrix, context_embedding)

        logits = self.qa_outputs(context_embedding_align)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MRCReader(BaseReader, ABC):
    def __init__(self, model, tokenizer, device):
        self.path_model_checkpoint = None
        self.train_dataset = None
        self.eval_dataset = None
        self.compute_metrics = None
        self.trainer = None
        self.device = device
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorCustom(tokenizer=self.tokenizer)
        self.compute_metrics = MRCEvaluator(tokenizer=self.tokenizer)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MRCQuestionAnsweringModel.from_pretrained(model_name_or_path).to(device)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        except:
            config_pretrained = load_json_data(os.path.join(model_name_or_path, 'config.json'))
            tokenizer = AutoTokenizer.from_pretrained(config_pretrained['_name_or_path'])
        return cls(model, tokenizer, device)

    def init_trainer(self, mrc_dataset: MRCDataset, **kwargs):
        self.path_model_checkpoint = kwargs.get(OUTPUT_DIR, 'model/qa')
        training_args = TrainingArguments(
            report_to='wandb',
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

        self.train_dataset = mrc_dataset.train_dataset
        self.eval_dataset = mrc_dataset.evaluator_dataset
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=mrc_dataset.train_dataset,
            eval_dataset=mrc_dataset.evaluator_dataset,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )

    def extract_answer(self, input_features, outputs):
        results = []
        for input_feature, start_logit, end_logit in zip(input_features, outputs.start_logits, outputs.end_logits):
            input_ids = input_feature[INPUT_IDS]
            words_length = input_feature[WORDS_LENGTH]
            answer_start_idx = sum(words_length[: torch.argmax(start_logit)])
            answer_end_idx = sum(words_length[: torch.argmax(end_logit) + 1])
            if answer_start_idx <= answer_end_idx:
                answer = self.tokenizer.convert_tokens_to_string(
                    self.tokenizer.convert_ids_to_tokens(input_ids[answer_start_idx:answer_end_idx])
                )
            else:
                answer = " "
            score_start = torch.max(torch.softmax(start_logit, dim=-1)).cpu().detach().numpy().tolist()
            score_end = torch.max(torch.softmax(end_logit, dim=-1)).cpu().detach().numpy().tolist()
            results.append({
                "answer": answer,
                "score_start": score_start,
                "score_end": score_end
            })
        return results

    def predict(self, queries: List[str], documents: List[List[Document]], **kwargs):
        logger.info(f'Number documents: {len(documents)}')
        assert len(queries) == len(documents), "Number question must equal number document"
        results = []
        for question, list_document in zip(queries, documents):
            results.append(
                self.qa_inference(
                    question=question,
                    documents=[
                        doc.document_context for doc in list_document
                    ]
                )
            )
            logger.info(results)
        return results

    def qa_inference(self, question: str, documents: List[str]):
        questions = [question] * len(documents)

        input_features_raw = make_input_feature_qa(
            questions=questions,
            documents=documents,
            tokenizer=self.tokenizer,
            max_length=368
        )
        input_features = self.data_collator(input_features_raw)
        for key, value in input_features.items():
            if isinstance(value, Tensor):
                input_features[key] = value.to(self.device)
        outs = self.model(**input_features)
        results = self.extract_answer(input_features_raw, outs)
        return results

    def train(self):
        wandb_api_key = os.getenv("WANDB_API")
        wandb.login(key=wandb_api_key)
        self.trainer.train()
        self.compute_metrics.log_predict = []  # refresh log
        self.evaluate(self.eval_dataset)

    def evaluate(self, dataset):
        self.trainer.evaluate(dataset)
        self.compute_metrics.save_log(path=f"{self.path_model_checkpoint}/log_predict.json")
