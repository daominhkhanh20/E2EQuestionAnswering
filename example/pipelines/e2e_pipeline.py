from e2eqavn.pipeline import E2EQuestionAnsweringPipeline
from e2eqavn.documents import Corpus
from e2eqavn.retrieval import *
from e2eqavn.mrc import * 
from e2eqavn.utils.io import load_yaml_file
import time

path_model = 'khanhbk20/vn-sentence-embedding'
config = load_yaml_file('config/train_random.yaml')
retrieval_config = config['retrieval']

corpus = Corpus.parser_uit_squad(**retrieval_config['data'])

bm25_retrieval = BM25Retrieval(corpus=corpus)
sbert_retrieval = SBertRetrieval.from_pretrained(model_name_or_path=path_model)
sbert_retrieval.update_embedding(corpus=corpus)
mrc_reader = MRCReader.from_pretrained('nguyenvulebinh/vi-mrc-large')
pipeline = E2EQuestionAnsweringPipeline(
    retrieval=[bm25_retrieval, sbert_retrieval],
    reader=mrc_reader
)

question = "Tên gọi nào được Phạm Văn Đồng sử dụng khi làm Phó chủ nhiệm cơ quan Biện sự xứ tại Quế Lâm?"
start_time = time.time()
result = pipeline.run(
    queries=question,
    top_k_bm25=50,
    top_k_sbert=3
)
print(result)
print(time.time() - start_time)