import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from keras.layers import Activation, Dense
from keras.models import Sequential, load_model
from web_processor import WebProcessor


class RelPredictor:
    def __init__(self, query, processor=None, model=None):
        self.query = query
        if model:
            self.model = model
        else:
            self.model = load_model('model/default_model.h5')
        if not processor:
            self.processor = WebProcessor(query=self.query)
        else:
            self.processor = processor

    def train_model(self, label, features):
        pass

    def save_model(self):
        self.model.save('model/default_model.h5')

    def get_relevance(self, url):
        self.processor.crawl_website(url)
        tag_text = self.processor.extract_by_tags()
        tf = self.processor.get_tfidf(tag_text)
        return self.model.predict(np.array([
            tf,
        ]))[0][0]


def main():
    query = 'artificial intelligence'.split()
    rp = RelPredictor(query=query)
    score = rp.get_relevance(
        'https://en.wikipedia.org/wiki/Artificial_intelligence')
    print(score)


if __name__ == '__main__':
    main()
