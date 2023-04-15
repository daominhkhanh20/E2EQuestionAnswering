from e2eqavn.retrieval import BM25Retrieval
from e2eqavn.documents import Corpus

from tqdm import tqdm

corpus = Corpus.parser_uit_squad(
    path_data='/media/dmk/D:/Data/Project/NLP/Thesis/data/UITSquad/test.json',
    mode_chunking=True,
    max_length=400,
    overlapping_size=80
)

bm25_retrieval = BM25Retrieval(
    corpus=corpus
)
list_top_k = [5, 10, 15, 50, 100, 150]
n_question = 0
list_hit_top_k = [0] * len(list_top_k)

for document in tqdm(corpus.list_document):
    document_id = document.document_id
    for question_answer in document.list_pair_question_answers:
        n_question += 1
        result = bm25_retrieval.retrieval(question_answer.question, top_k=list_top_k[-1])
        list_document_id = [doc.document_id for doc in result]
        for idx, top_k in enumerate(list_top_k):
            if document_id in list_document_id[-top_k:]:
                list_hit_top_k[idx] += 1

print([value/n_question for value in list_hit_top_k])