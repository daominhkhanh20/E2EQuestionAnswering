from e2eqavn.pipeline import E2EQuestionAnsweringPipeline
from e2eqavn.documents import Corpus
from e2eqavn.retrieval import *
from e2eqavn.mrc import * 
from e2eqavn.utils.io import load_yaml_file
import time
import pprint

pp = pprint.PrettyPrinter(depth=4)

path_model = 'khanhbk20/vn-sentence-embedding'
config_pipeline = load_yaml_file('config/train_qa.yaml')
corpus = Corpus.parser_uit_squad(
        config_pipeline[DATA][PATH_EVALUATOR],
        **config_pipeline.get(CONFIG_DATA, {})
    )

bm25_retrieval = BM25Retrieval(corpus=corpus)
sbert_retrieval = SBertRetrieval.from_pretrained(model_name_or_path=path_model)
sbert_retrieval.update_embedding(corpus=corpus)
mrc_reader = MRCReader.from_pretrained('model/qa/checkpoint-1144')
pipeline = E2EQuestionAnsweringPipeline(
    retrieval=[bm25_retrieval, sbert_retrieval],
    reader=mrc_reader
)

questions = [
    "Vị trí địa lý của Paris có gì đặc biệt?"
    ]
start_time = time.time()
result = pipeline.run(
    queries=questions,
    top_k_bm25=50,
    top_k_sbert=3,
    top_k_qa=1
)
print(result)
predictions = []
for idx, ans_pred in enumerate(result['answer']):
    predictions.append(
        {'prediction_text': ans_pred[0]['answer'], 'id': str(idx)}
    )
print(predictions)
