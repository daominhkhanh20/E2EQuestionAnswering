import os.path

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
        preds = list(zip(logits[0], logits[1]))
        truth_labels, span_answer_ids, input_ids, word_lengths = list(zip(labels[0], labels[1])), labels[2], labels[3], labels[4]
        predictions, references = [], []
        for idx, (predict, span_answer_id, input_id, word_length) in enumerate(
            list(zip(preds, span_answer_ids, input_ids, word_lengths))
        ):
            span_answer_id = np.delete(span_answer_id, np.where(span_answer_id == -100))
            start_pred = sum(word_length[: np.argmax(predict[0])])
            end_pred = sum(word_length[: np.argmax(predict[1]) + 1])
            answer_pred = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(input_id[start_pred: end_pred])
            )

            answer_truth = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(span_answer_id)
            )
            self.log_predict.append({
                "pred": answer_pred,
                "truth": answer_truth
            })

            predictions.append({'prediction_text': answer_pred, 'id': str(idx)})
            references.append({'answers': {'answer_start': [start_pred], 'text': [answer_truth]}, 'id': str(idx)})
        return self.metric.compute(predictions=predictions, references=references)

    def save_log(self, path: str = 'log/result.json'):
        path_folder = path.rsplit('/', 1)[0]
        if not os.path.exists(path_folder):
            os.makedirs(path_folder, exist_ok=True)
        write_json_file(self.log_predict, path)
        self.log_predict = []


