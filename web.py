import numpy as np
import torch.nn as nn

from web_processor import WebProcessor
from rel_predictor import RelPredictor
from link_evaluation import LinkEvaluation
from link_evaluation import CNN

class Web:
    def __init__(self, query, W2V):
        self.processor = WebProcessor(query, model=W2V)
        self.page_eval = RelPredictor(query, self.processor)
        self.outlink_eval = LinkEvaluation(query, self.processor)

    def page_target_topics(self, link):
        score = self.page_eval.get_relevance(link)
        score = np.clip(score, 0, 1)
        return score

    def outlink_target_topics(self, relevant_urls):
        score = self.outlink_eval.get_link_score(relevant_urls)
        for i in score:
            score[i] = np.clip(score[i], 0, 1)
        return score
