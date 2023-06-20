from e2eqavn.pipeline import E2EQuestionAnsweringPipeline
from e2eqavn.documents import Corpus
from e2eqavn.retrieval import *
from e2eqavn.utils.io import load_yaml_file

path_model = 'khanhbk20/vn-sentence-embedding'
config = load_yaml_file('config/train_qa.yaml')

corpus = Corpus.parser_uit_squad(path_data=config['data']['path_evaluator'])

bm25_retrieval = BM25Retrieval(corpus=corpus)
sbert_retrieval = SBertRetrieval.from_pretrained(model_name_or_path=path_model)
sbert_retrieval.update_embedding(corpus=corpus)
pipeline = E2EQuestionAnsweringPipeline(
    retrieval=[bm25_retrieval, sbert_retrieval]
)

question = "Vị trí địa lý của Paris có gì đặc biệt?"
result = pipeline.run(
    queries=question,
    top_k_bm25=50,
    top_k_sbert=3
)
for doc in result['documents'][0]:
    print(doc.document_context)
