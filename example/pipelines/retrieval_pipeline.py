from e2eqavn.pipeline import E2EQuestionAnsweringPipeline
from e2eqavn.documents import Corpus
from e2eqavn.retrieval import *
from e2eqavn.utils.io import load_yaml_file


path_model = 'model/Model'
config = load_yaml_file('config/train_random.yaml')
retrieval_config = config['retrieval']

corpus = Corpus.parser_uit_squad(**retrieval_config['data'])

bm25_retrieval = BM25Retrieval(corpus=corpus)
sbert_retrieval = SBertRetrieval.from_pretrained(model_name_or_path=path_model)
sbert_retrieval.corpus = corpus
# sbert_retrieval.update_embedding(corpus=corpus)

pipeline = E2EQuestionAnsweringPipeline(
    retrieval=[bm25_retrieval, sbert_retrieval]
)

question = "Tên gọi nào được Phạm Văn Đồng sử dụng khi làm Phó chủ nhiệm cơ quan Biện sự xứ tại Quế Lâm?"
result = pipeline.run(
    query=question,
    top_k_bm25=50,
    top_k_sbert=1
)
for doc in result['documents']:
    print(doc.document_context)