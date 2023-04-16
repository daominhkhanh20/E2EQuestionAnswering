from e2eqavn.retrieval import BM25Retrieval
from e2eqavn.documents import Corpus
from e2eqavn.pipeline import E2EQuestionAnsweringPipeline
from e2eqavn.documents import Corpus
from e2eqavn.utils.io import load_yaml_file
from tqdm import tqdm

config = load_yaml_file('config/infer.yaml')
retrieval_config = config['retrieval']
corpus = Corpus.parser_uit_squad(**retrieval_config['data'])
print(len(corpus.list_document))

bm25_retrieval = BM25Retrieval(corpus=corpus)
question = "Sự kiện nào đánh dấu sự hình thành của lưu vực sông Loire và sông Seine?"
result = bm25_retrieval.retrieval([question], top_k=5)[0]
for doc in result:
    print(doc.document_context, doc.bm25_score)
    print('\n\n')
# list_top_k = [5, 10, 15, 50, 100, 150]
# n_question = 0
# list_hit_top_k = [0] * len(list_top_k)
# for document in tqdm(corpus.list_document):
#     if len(document.list_pair_question_answers) == 0:
#         continue
#     document_id = document.document_id
#     for question_answer in document.list_pair_question_answers:
#         n_question += 1
#         result = bm25_retrieval.retrieval([question_answer.question], top_k=list_top_k[-1])[0]
#         list_document_id = [doc.document_id for doc in result]
#         for idx, top_k in enumerate(list_top_k):
#             if document_id in list_document_id[-top_k:]:
#                 list_hit_top_k[idx] += 1
#
# print([value/n_question for value in list_hit_top_k])