import numpy as np
import re

from web_processor import WebProcessor
from rel_predictor import RelPredictor
from link_evaluation import LinkEvaluation

class Web:
    def __init__(self, topics):
        self.query = re.split(' |,', topics)
        self.processor = WebProcessor(self.query)
        self.page_eval = RelPredictor(self.query, self.processor)
        self.outlink_eval = LinkEvaluation(self.query, self.processor)

    def page_target_topics(self, link):
        score = self.page_eval.get_relevance(link)
        score = np.clip(score, 0, 1)
        return score

    def outlink_target_topics(self, relevant_urls):
        score = self.outlink_eval.get_link_score(relevant_urls)
        for i in score:
            score[i] = np.clip(score[i], 0, 1)
        return score
