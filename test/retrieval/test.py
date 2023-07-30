from e2eqavn.retrieval import BM25Retrieval
from e2eqavn.processor import BM25Scoring
from e2eqavn.documents import Corpus
from e2eqavn.pipeline import E2EQuestionAnsweringPipeline
import hashlib
from e2eqavn.evaluate import InformationRetrievalEvaluatorCustom
from e2eqavn.utils.calculate import make_input_for_retrieval_evaluator
from tqdm import tqdm
from e2eqavn.utils.io import load_json_data
#
corpus = Corpus.init_corpus(
    path_data='/media/dmk/D:/Data/Project/NLP/E2EQuestionAnswering/corpus.json',
)
# print(corpus.list_document[0].document_context)
# # print(corpus.list_document_context)
# # print(len(corpus.list_document))
# bm25_retrieval = BM25Retrieval(
#     corpus=corpus
# )
question = 'Lý thuyết thông tin thuật toán là gì?'
# # print(bm25_retrieval.bm25_model.get_top_k(query=question, top_k=4))
# result = bm25_retrieval.retrieval(queries=question, top_k=10)
# for doc in result[0]:
#     print(f"{doc.document_context}\n\n")

scoring = BM25Scoring(corpus=[doc.document_context for doc in corpus.list_document])
print(scoring.get_top_k('Lý thuyết thông tin là là gì?', top_k=15))