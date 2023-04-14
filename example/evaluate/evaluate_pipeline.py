from e2eqavn.pipeline import E2EQuestionAnsweringPipeline
from e2eqavn.documents import Corpus
from e2eqavn.evaluate import InformationRetrievalEvaluatorCustom
import hashlib
from e2eqavn.retrieval import *
from e2eqavn.utils.io import load_yaml_file


path_model = '/kaggle/input/model-temp/Model'
config = load_yaml_file('config/infer.yaml')
retrieval_config = config['retrieval']

corpus = Corpus.parser_uit_squad(**retrieval_config['data'])

bm25_retrieval = BM25Retrieval(corpus=corpus)
sbert_retrieval = SBertRetrieval.from_pretrained(model_name_or_path=path_model)
sbert_retrieval.update_embedding(corpus=corpus)

pipeline = E2EQuestionAnsweringPipeline(
    retrieval=[bm25_retrieval, sbert_retrieval]
)

context_copurs = {doc.document_id: doc.document_context for doc in corpus.list_document}
queries = {}
relevant_docs = {}
for doc in corpus.list_document:
    # if doc.document_id not in relevant_docs:
    #     relevant_docs[doc.document_id] = set()
    for question_answer in doc.list_pair_question_answers:
        ques_id = hashlib.sha1(str(question_answer.question).encode('utf-8')).hexdigest()
        queries[ques_id] = question_answer.question
        if ques_id not in relevant_docs:
            relevant_docs[ques_id] = set()
        relevant_docs[ques_id].add(doc.document_id)

evaluator = InformationRetrievalEvaluatorCustom(
    corpus=context_copurs,
    queries=queries,
    relevant_docs=relevant_docs
)
scores = evaluator.compute_metrices_retrieval(pipeline=pipeline)
print(scores)