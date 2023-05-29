from e2eqavn.documents import Corpus
from e2eqavn.utils.io import load_yaml_file

config = load_yaml_file('config/infer.yaml')
retrieval_config = config['retrieval']
print(retrieval_config['data'])
corpus = Corpus.parser_uit_squad(
    **retrieval_config['data']
)

# corpus_dict = load_json_data('/media/dmk/D:/Data/Project/NLP/Thesis/data/UITSquad/sample.json')
# corpus = Corpus.init_corpus(corpus=corpus_dict)

# print(corpus.list_document[10].index)
print(f"Number documents {len(corpus.list_document)}")
print(f"Number question: {corpus.n_pair_question_answer}")
corpus.save_corpus('/home/dmk/Documents/ThesisDeploy/ThesisDeploy/model/bm25_retrieval/1/corpus.json')
