from e2eqavn.documents import Corpus
from e2eqavn.utils.io import load_yaml_file

config = load_yaml_file('config/test_load_corpus.yaml')
corpus = Corpus.init_corpus(
    path_data=config['data']['path_train'],
    **config['config_data']
)

# corpus_dict = load_json_data('/media/dmk/D:/Data/Project/NLP/Thesis/data/UITSquad/sample.json')
# corpus = Corpus.init_corpus(corpus=corpus_dict)

# print(corpus.list_document[10].index)
print(f"Number documents {len(corpus.list_document)}")
print(f"Number question: {corpus.n_pair_question_answer}")
