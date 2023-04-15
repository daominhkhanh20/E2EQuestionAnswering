from e2eqavn.retrieval import BM25Retrieval
from e2eqavn.documents import Corpus
import argparse
import time
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool


parser = argparse.ArgumentParser()
parser.add_argument('--mode_parallel', default=False, type=lambda x: x.lower() == 'true')
args = parser.parse_args()
corpus = Corpus.parser_uit_squad(
    path_data='/media/dmk/D:/Data/Project/NLP/Thesis/data/UITSquad/train.json',
    mode_chunking=True,
    max_length=400,
    overlapping_size=80
)

bm25_retrieval = BM25Retrieval(
    corpus=corpus
)

list_questions = []
for doc in corpus.list_document:
    for question_answer in doc.list_pair_question_answers:
        list_questions.append(question_answer.question)

results = []
print(len(list_questions))
list_questions = list_questions[:500]
start_time = time.time()
if not args.mode_parallel:
    for question in list_questions:
        top_k_index = bm25_retrieval.retrieval(question, top_k=10)
        results.append(top_k_index)
else:
    # mp.set_start_method('fork')
    results = bm25_retrieval.retrieval(list_questions)
print(time.time() - start_time)

