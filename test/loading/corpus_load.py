from e2eqavn.documents import Corpus
from e2eqavn.utils.io import load_json_data
corpus = Corpus.parser_uit_squad(
    path_data='/media/dmk/D:/Data/Project/NLP/Thesis/data/UITSquad/train.json',
    mode_chunking=True,
    max_length=250,
    overlapping_size=60
)

# corpus_dict = load_json_data('/media/dmk/D:/Data/Project/NLP/Thesis/data/UITSquad/sample.json')
# corpus = Corpus.init_corpus(corpus=corpus_dict)

# print(corpus.list_document[10].index)
print(f"Number documents {len(corpus.list_document)}")
print(f"Number question: {corpus.n_pair_question_answer}")
corpus.save_corpus('corpus.json')