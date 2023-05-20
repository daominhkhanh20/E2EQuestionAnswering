from typing import *
import logging
from tqdm import trange
import logging
import pandas as pd
from sentence_transformers.util import cos_sim, dot_score

from e2eqavn.retrieval import *
from e2eqavn.keywords import *
# from e2eqavn.pipeline import E2EQuestionAnsweringPipeline
from sentence_transformers.evaluation import InformationRetrievalEvaluator

logger = logging.getLogger(__name__)


class InformationRetrievalEvaluatorCustom(InformationRetrievalEvaluator):
    def __init__(self, queries: Dict[str, str], corpus: Dict[str, str], relevant_docs: Dict[str, Set[str]],
                 corpus_chunk_size: int = 50000, mrr_at_k: List[int] = [10], ndcg_at_k: List[int] = [10],
                 accuracy_at_k: List[int] = [1, 3, 5, 10], precision_recall_at_k: List[int] = [1, 3, 5, 10],
                 map_at_k: List[int] = [10], show_progress_bar: bool = False, batch_size: int = 32, name: str = '',
                 write_csv: bool = True,
                 score_functions: List[Callable[[Tensor, Tensor], Tensor]] = {'cos_sim': cos_sim,
                                                                              'dot_score': dot_score},
                 main_score_function: str = None):
        super().__init__(queries, corpus, relevant_docs, corpus_chunk_size, mrr_at_k, ndcg_at_k, accuracy_at_k,
                         precision_recall_at_k, map_at_k, show_progress_bar, batch_size, name, write_csv,
                         score_functions, main_score_function)
        self.queries = list(queries.values())

    def compute_metrices_retrieval(self, pipeline,
                                   **kwargs) -> Dict[str, float]:
        top_k_bm25 = kwargs.get(TOP_K_BM25, 30)
        get_scoring_method = kwargs.get(GET_SCORE, EMBEDDING_SCORE)
        if kwargs.get(TOP_K_SBERT, None):
            top_k_sbert = kwargs.get(TOP_K_SBERT)
        else:
            top_k_sbert = max(max(self.mrr_at_k), max(self.ndcg_at_k), max(self.accuracy_at_k),
                              max(self.precision_recall_at_k),
                              max(self.map_at_k))
        list_question = self.queries
        query_result_list = [[] for _ in range(len(self.queries))]

        results = pipeline.run(
            queries=list_question,
            top_k_bm25=top_k_bm25,
            top_k_sbert=top_k_sbert
        )['documents']

        assert len(results) == len(list_question), "Number result not equal number question"

        for query_iter in range(len(list_question)):
            for doc in results[query_iter]:
                if get_scoring_method == EMBEDDING_SCORE:
                    query_result_list[query_iter].append({'corpus_id': doc.document_id,
                                                          'score': doc.embedding_similarity_score})
                elif get_scoring_method == COMBINE_BM25_EMBED_SCORE:
                    query_result_list[query_iter].append({'corpus_id': doc.document_id,
                                                          'score': doc.score})
        scores = self.compute_metrics(query_result_list)
        if kwargs.get(SAVE_LOG, True):
            df = pd.DataFrame.from_dict(scores).rename_axis('top_k').reset_index()
            path_save = kwargs.get(PATH_SAVE_LOG, 'retrieval_accurate.csv')
            df.to_csv(path_save, index=False)
            logger.info(f"Save retrieval evaluate log at {path_save}")
        return scores
