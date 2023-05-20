from e2eqavn.utils.io import load_yaml_file
from e2eqavn.documents import Corpus
from e2eqavn.datasets import TripletDataset
from e2eqavn.processor import RetrievalGeneration
from e2eqavn.retrieval import SentenceBertLearner
from e2eqavn.utils.calculate import make_vnsquad_retrieval_evaluator
from e2eqavn.keywords import  *

config_pipeline = load_yaml_file('config/train_qa.yaml')
train_corpus = Corpus.parser_uit_squad(
    config_pipeline[DATA][PATH_TRAIN],
    **config_pipeline.get(CONFIG_DATA, {})
)
retrieval_config = config_pipeline.get(RETRIEVAL, None)
reader_config = config_pipeline.get(READER, None)
if retrieval_config:
    retrieval_sample = RetrievalGeneration.generate_sampling(train_corpus, **retrieval_config[PARAMETERS])
    train_dataset = TripletDataset.load_from_retrieval_sampling(retrieval_sample=retrieval_sample)
    dev_evaluator = make_vnsquad_retrieval_evaluator(
        path_data_json=config_pipeline[DATA][PATH_EVALUATOR]
    )

    retrieval_learner = SentenceBertLearner.from_pretrained(
        model_name_or_path=retrieval_config[MODEL].get(MODEL_NAME_OR_PATH, 'khanhbk20/vn-sentence-embedding')
    )
    retrieval_learner.train(
        train_dataset=train_dataset,
        loss_fn_config=retrieval_config[MODEL].get(LOSS_FN, None),
        dev_evaluator=dev_evaluator,
        batch_size=retrieval_config[MODEL].get(BATCH_SIZE, 16),
        epochs=retrieval_config[MODEL].get(EPOCHS, 1)
    )
