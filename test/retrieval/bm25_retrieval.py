from e2eqavn.retrieval import BM25Retrieval
from e2eqavn.documents import Corpus
from e2eqavn.pipeline import E2EQuestionAnsweringPipeline
import hashlib
from e2eqavn.evaluate import InformationRetrievalEvaluatorCustom
from e2eqavn.utils.calculate import make_input_for_retrieval_evaluator
from tqdm import tqdm

corpus = Corpus.init_corpus(
    path_data='data/hand/edu_test.json',
)
# print(corpus.list_document_context)
# print(len(corpus.list_document))
bm25_retrieval = BM25Retrieval(
    corpus=corpus
)
# question = 'Đề tài nghiên cứu khoa học là gì?'
# result = bm25_retrieval.retrieval(queries=question, top_k=3)
# for doc in result[0]:
#     print(doc.document_context)

pipeline = E2EQuestionAnsweringPipeline(
    retrieval=[bm25_retrieval]
)
context_copurs, queries, relevant_docs = make_input_for_retrieval_evaluator(
        path_data_json='data/hand/edu_test.json'
    )
# print(context_copurs)
context_copurs1 = {doc.document_id: doc.document_context for doc in corpus.list_document}
queries1 = {}
relevant_docs1 = {}
for doc in corpus.list_document:
    if len(doc.list_pair_question_answers) == 0:
        continue
    for question_answer in doc.list_pair_question_answers:
        ques_id = hashlib.sha1(str(question_answer.question).encode('utf-8')).hexdigest()
        queries1[ques_id] = question_answer.question
        if ques_id not in relevant_docs1:
            relevant_docs1[ques_id] = set()
        relevant_docs1[ques_id].add(doc.document_id)

print(len(context_copurs1), len(context_copurs))
print(len(queries1), len(queries))
print(len(relevant_docs1), len(relevant_docs))
evaluator = InformationRetrievalEvaluatorCustom(
    corpus=context_copurs,
    queries=queries,
    relevant_docs=relevant_docs
)
scores = evaluator.compute_metrices_retrieval(pipeline=pipeline, get_score='combine_bm25_embed_score')
print(scores)
#
# list_top_k = [5, 10, 15, 50, 100, 150]
# n_question = 0
# list_hit_top_k = [0] * len(list_top_k)
#
# for document in tqdm(corpus.list_document):
#     document_id = document.document_id
#     for question_answer in document.list_pair_question_answers:
#         n_question += 1
#         result = bm25_retrieval.retrieval(question_answer.question, top_k=list_top_k[-1])[0]
#         list_document_id = [doc.document_id for doc in result]
#         for idx, top_k in enumerate(list_top_k):
#             if document_id in list_document_id[-top_k:]:
#                 list_hit_top_k[idx] += 1
#
# print([value/n_question for value in list_hit_top_k])