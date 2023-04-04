from src.documents import Corpus

corpus = Corpus.parser_uit_squad(
    path_file='/media/dmk/D:/Data/Project/NLP/Thesis/data/UITSquad/train.json',
    mode_chunking=True,
    max_length=400,
    overlapping_size=80
)
print(f"Number pair question and answer: {len(corpus.list_document)}")
print(f"Number question: {corpus.n_pair_question_answer}")
print(corpus.list_document[0])
corpus.save_corpus('corpus.json')