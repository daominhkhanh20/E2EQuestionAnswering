from e2eqavn.documents import Corpus
from e2eqavn.datasets import MRCDataset
from e2eqavn.utils.io import load_yaml_file, write_json_file
from e2eqavn.keywords import *
from e2eqavn.mrc import MRCReader
import wandb
import os


config_pipeline = load_yaml_file('config/train_qa1.yaml')
train_corpus = Corpus.init_corpus(
        config_pipeline[DATA][PATH_TRAIN],
        **config_pipeline.get(CONFIG_DATA, {})
    )
# retrieval_config = config_pipeline.get(RETRIEVAL, None)
reader_config = config_pipeline.get(READER, None)
eval_corpus = Corpus.init_corpus(
            config_pipeline[DATA][PATH_EVALUATOR],
            **config_pipeline.get(CONFIG_DATA, {})
        )
mrc_dataset = MRCDataset.init_mrc_dataset(
    corpus_train=train_corpus,
    corpus_eval=eval_corpus,
    model_name_or_path=reader_config[MODEL].get(MODEL_NAME_OR_PATH, 'khanhbk20/mrc_testing'),
    max_length=reader_config[MODEL].get(MAX_LENGTH, 368),
    **reader_config.get(DATA_ARGUMENT, {})
)
reader_model = MRCReader.from_pretrained(
    model_name_or_path=reader_config[MODEL].get(MODEL_NAME_OR_PATH, 'khanhbk20/mrc_testing'),
    lambda_weight=reader_config.get(DATA_ARGUMENT, {}).get(LAMBDA_WEIGHT, 0.6)
)
reader_model.init_trainer(mrc_dataset=mrc_dataset, **reader_config[MODEL])
reader_model.train()
# loader = reader_model.trainer.get_train_dataloader()
# sample = next(iter(loader))
# outs = reader_model.model(**sample)
# print(outs.keys())


# train_corpus = Corpus.parser_uit_squad(config_qa['data']['path_train'])
# eval_corpus = Corpus.parser_uit_squad(config_qa['data']['path_evaluator'])
# if not os.path.exists(config_qa['model'].get(OUTPUT_DIR, 'model/qa')):
#     os.makedirs(config_qa['model'].get(OUTPUT_DIR, 'model/qa'))
# write_json_file(config_qa, os.path.join(config_qa['model'].get(OUTPUT_DIR, 'model/qa'), 'parameter.json'))

# config = {**config_qa['parameters'], **config_qa['model']}
# wandb.init(project='E2E_QA_THESISV2', config=config)
# train_corpus = Corpus.parser_uit_squad(config['data']['path_train'], **config.get('config_qa', {}))
# eval_corpus = Corpus.parser_uit_squad(config['data']['path_evaluator'], **config.get('config_qa', {}))
# dataset = MRCDataset.init_mrc_dataset(
#     corpus_train=train_corpus,
#     corpus_eval=eval_corpus,
#     model_name_or_path=config_qa['model'][MODEL_NAME_OR_PATH],
#     max_length=config['config_data'].get('max_length', config_qa['model'].get('max_length', 400)),
#     mode_chunking=config['config_data'].get('mode_chunking', False)
# )
# print(len(dataset.train_dataset))
# print(len(dataset.evaluator_dataset))
# reader = MRCReader.from_pretrained(config_qa['model'][MODEL_NAME_OR_PATH])
# reader.init_trainer(dataset, **config_qa['model'])
# # reader.train()