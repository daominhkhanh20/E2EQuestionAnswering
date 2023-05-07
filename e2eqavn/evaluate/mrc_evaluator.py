import numpy as np
from datasets import load_metric
from e2eqavn.utils.io import write_json_file


class MRCEvaluator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.metric = load_metric('squad')
        self.log_predict = []

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
            self.log_predict.append({
                "pred": answer_pred,
                "truth": answer_truth
            })

            predictions.append({'prediction_text': answer_pred, 'id': str(idx)})
            references.append({'answers': {'answer_start': [start_position], 'text': [answer_truth]}, 'id': str(idx)})
        return self.metric.compute(predictions=predictions, references=references)

    def save_log(self, path: str = 'log/result.json'):
        write_json_file(self.log_predict, path)
        self.log_predict = []


