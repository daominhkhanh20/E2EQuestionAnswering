from e2eqavn.utils.io import load_yaml_file
from e2eqavn.documents import Corpus
from e2eqavn.datasets import TripletDataset
from e2eqavn.processor import RetrievalGeneration
from e2eqavn.retrieval import SentenceBertLearner
from e2eqavn.utils.calculate import make_vnsquad_retrieval_evaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss

config = load_yaml_file('config/train_qa.yaml')
retrieval_config = config['retrieval']
corpus = Corpus.parser_uit_squad(path_data=config['data']['path_train'],
                                 **config.get('config_data', {}))
# corpus.save_corpus('corpus.json')
retrieval_sampling = RetrievalGeneration.generate_sampling(corpus, **retrieval_config['parameters'])
train_dataset = TripletDataset.load_from_retrieval_sampling(retrieval_sampling)
dev_evaluator = make_vnsquad_retrieval_evaluator(
    path_data_json=retrieval_config['data']['path_evaluator']
)

learner = SentenceBertLearner.from_pretrained(
    retrieval_config['model']['model_name_or_path']
)

learner.train(
    train_dataset=train_dataset,
    loss_fn_config=retrieval_config['model']['loss_fn'],
    dev_evaluator=dev_evaluator,
    batch_size=retrieval_config['model']['batch_size'],
    epochs=retrieval_config['model']['epochs']
)
