from e2eqavn.pipeline import E2EQuestionAnsweringPipeline
from e2eqavn.documents import Corpus
from e2eqavn.evaluate import InformationRetrievalEvaluatorCustom
import hashlib
from e2eqavn.retrieval import *
from e2eqavn.keywords import *
from e2eqavn.utils.io import load_yaml_file
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--question', type=str)
parser.add_argument('--top_k_sbert', type=int, default=3)
parser.add_argument('--get_score', type=str, default='combine_bm25_embed_score')
args = parser.parse_args()


path_model = 'vn-sentence-embedding'
config = load_yaml_file('config/infer.yaml')
retrieval_config = config['retrieval']

corpus = Corpus.parser_uit_squad(**retrieval_config['data'])
# print(len(corpus.list_document))

bm25_retrieval = BM25Retrieval(corpus=corpus)
# sbert_retrieval = SBertRetrieval.from_pretrained(model_name_or_path=path_model)
# sbert_retrieval.update_embedding(corpus=corpus, path_corpus_embedding='corpus_embedding.pth')

pipeline = E2EQuestionAnsweringPipeline(
    retrieval=[bm25_retrieval]
)
list_question = []
context_copurs = {doc.document_id: doc.document_context for doc in corpus.list_document}
queries = {}
relevant_docs = {}
for doc in corpus.list_document:
    if len(doc.list_pair_question_answers) == 0:
        continue
    for question_answer in doc.list_pair_question_answers:
        ques_id = hashlib.sha1(str(question_answer.question).encode('utf-8')).hexdigest()
        queries[ques_id] = question_answer.question
        if ques_id not in relevant_docs:
            relevant_docs[ques_id] = set()
        relevant_docs[ques_id].add(doc.document_id)

# normal_indexs = []
i = 0
# for question_id, question in tqdm(queries.items()):
#     result = bm25_retrieval.retrieval(question, top_k=5)[0]
#     list_id = [doc.document_id for doc in result]
#     if list(relevant_docs[question_id])[0] in list_id:
#         i += 1

# print(i/len(queries))


# batching_indexs = []
# result = pipeline.run(queries=list(queries.values()), top_k_bm25=5)['documents']
# for index, question_id in enumerate(queries.keys()):
#     list_id = [doc.document_id for doc in result[index]]
#     if list(relevant_docs[question_id])[0] in list_id:
#         i += 1
# print(i)
# print(len(queries))
# print(i/len(queries))
# for i in range(len(result)):
#     list_idx = [doc.index for doc in result[i]]
#     batching_indexs.append(list_idx)
#
# for i in range(len(list_question)):
#     if batching_indexs[i] != normal_indexs[i]:
#         print(1)
# question = [args.question]
# result = pipeline.run(
#     queries=question,
#     top_k_bm25=20,
#     top_k_sbert=args.top_k_sbert
# )['documents']
# for doc in result[0]:
#     print(doc.document_context, doc.bm25_score, doc.embedding_similarity_score)
#     print("\n\n")

# context_copurs = {doc.document_id: doc.document_context for doc in corpus.list_document}
# queries = {}
# relevant_docs = {}
# for doc in corpus.list_document:
#     if len(doc.list_pair_question_answers) == 0:
#         continue
#     for question_answer in doc.list_pair_question_answers:
#         ques_id = hashlib.sha1(str(question_answer.question).encode('utf-8')).hexdigest()
#         queries[ques_id] = question_answer.question
#         if ques_id not in relevant_docs:
#             relevant_docs[ques_id] = set()
#         relevant_docs[ques_id].add(doc.document_id)
#
evaluator = InformationRetrievalEvaluatorCustom(
    corpus=context_copurs,
    queries=queries,
    relevant_docs=relevant_docs
)
scores = evaluator.compute_metrices_retrieval(pipeline=pipeline, get_score=args.get_score)
print(scores)