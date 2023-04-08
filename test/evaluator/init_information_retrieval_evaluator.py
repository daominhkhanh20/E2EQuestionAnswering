from e2eqavn.utils.calculate import make_vnsquad_retrieval_evaluator

evaluator = make_vnsquad_retrieval_evaluator(
    path_data_json='data/UITSquad/dev.json'
)