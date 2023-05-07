import numpy as np
from datasets import load_metric


class MRCEvaluator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.metric = load_metric('squad')

    def __call__(self, eval_pred):
        logits, labels = eval_pred
        start_logits, end_logits = logits[0], logits[1]
        start_positions, end_positions, input_ids = labels[0], labels[1], labels[2]
        predictions, references = [], []
        for idx, (start_logit, end_logit, start_position, end_position, input_id) in enumerate(
            list(zip(start_logits, end_logits, start_positions, end_positions, input_ids))
        ):
            start_idx_pred = np.argmax(start_logit)
            end_idx_pred = np.argmax(end_logit)
            answer_pred = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(input_id[start_idx_pred: end_idx_pred + 1])
            )

            answer_truth = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(input_id[start_position: end_position + 1])
            )

            predictions.append({'prediction_text': answer_pred, 'id': str(idx)})
            references.append({'answers': {'answer_start': [start_position], 'text': [answer_truth]}, 'id': str(idx)})
        return self.metric.compute(predictions=predictions, references=references)

