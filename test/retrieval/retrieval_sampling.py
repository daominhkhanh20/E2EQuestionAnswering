from e2eqavn.documents import Corpus
from e2eqavn.processor import RetrievalGeneration
from e2eqavn.utils.io import load_json_data, load_yaml_file
config_pipeline = load_yaml_file('config/train.yaml')

retrieval_config = config_pipeline['pipeline']['retrieval']
if retrieval_config['is_train']:
    corpus = Corpus.parser_uit_squad(
        path_file=retrieval_config['data']['path_data'],
        mode_chunking=retrieval_config['data']['mode_chunking'],
        max_length=retrieval_config['data']['max_length'],
        overlapping_size=retrieval_config['data']['overlapping_size']
    )
    print(corpus.n_pair_question_answer)
    sample = RetrievalGeneration.generate_sampling(corpus=corpus, **retrieval_config['data'])
    print(len(sample.list_retrieval_sample))