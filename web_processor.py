import os
import re
import string
import urllib.parse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup

import gensim
from keras.layers import (Activation, Conv1D, Dense, Dropout, Embedding,
                          GlobalMaxPooling1D)
from keras.models import Sequential
from keras.preprocessing import sequence


class WebProcessor:
    def __init__(self, query, tags=None, model=None, sim_threshold=0.4):
        self.query = query
        if tags:
            self.tags = tags
        else:
            self.tags = [
                'title', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol',
                'table'
            ]
        if model:
            self.model = model
        else:
            path = '/mnt/d/Word2Vec/GoogleNews-vectors-negative300.bin'
            if os.path.isfile(path):
                self.model = gensim.models.KeyedVectors.load_word2vec_format(
                    path, binary=True)
                print('Model loaded.')
            else:
                raise FileNotFoundError(
                    f"No such file: '{path}'\n"
                    "Pre-trained word and phrase vectors not found. "
                    "You can download the file at "
                    "https://code.google.com/archive/p/word2vec/.")
        self.sim_threshold = sim_threshold

    def get_topic_relevance(self):
        pass

    def crawl_website(self, url):
        self.url = url
        try:
            r = requests.get(url)
            r.raise_for_status()
        except Exception:
            return None
        self.soup = BeautifulSoup(r.text, 'html.parser')
        # print('Web page loaded..')

    def extract_by_tags(self):
        tag_text = defaultdict(str)
        for tag in self.tags:
            for elem in self.soup.find_all(tag):
                stripped = elem.text.translate(
                    str.maketrans(
                        str.maketrans(dict.fromkeys(string.punctuation))))
                stripped = re.sub(r'\s+', ' ', stripped).strip()
                if stripped:
                    tag_text[tag] += f' {stripped.lower()}'.strip()
        return tag_text

    def get_hyperlinks(self):
        self.links = {}
        for link in self.soup.find_all('a'):
            url = urllib.parse.urljoin(self.url, link.get('href'))
            self.links[url] = ' '.join(
                [self.links.get(url, '') + link.text.strip()])
        return self.links

    def get_tf(self, text):
        if not text:
            return 0
        words = text.split()
        num_rel_words = np.zeros(len(self.query))
        for word in words:
            for idx, topic in enumerate(self.query):
                try:
                    sim = self.model.similarity(topic, word)
                    # sim = np.random.random()
                    if sim >= self.sim_threshold:
                        num_rel_words[idx] += 1
                except KeyError:
                    pass
        return np.max(num_rel_words) / len(text)

    def get_tfidf(self, tag_text):
        tf = np.empty(len(self.tags))
        for idx, tag in enumerate(self.tags):
            try:
                tf[idx] = self.get_tf(tag_text[tag])
            except KeyError:
                tf[idx] = 0
        return tf


def train_cnn(filename, query):
    wp = WebProcessor(query=query)
    with open(filename) as f:
        lines = f.read().splitlines()
    lines = lines[:20]
    data = list(map(lambda x: (float(x.split()[0]), x.split()[1]), lines))
    label = np.empty(len(data))
    features = np.empty([len(data), len(wp.tags)])
    for idx, (score, url) in enumerate(data):
        label[idx] = score
        wp.crawl_website(url=url)
        tag_text = wp.extract_by_tags()
        features[idx] = wp.get_tfidf(tag_text).flatten()
    # print(label)
    # print(features)

    train_size = int(len(data) * .8)
    print("Train size: %d" % train_size)
    print("Test size: %d" % (len(data) - train_size))
    x_train = features[:train_size]
    y_train = label[:train_size]
    x_test = features[train_size:]
    y_test = label[train_size:]

    epochs = 2
    batch_size = 32

    model = Sequential()
    model.add(
        Dense(
            12,
            input_dim=features.shape[1],
            kernel_initializer='normal',
            activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
    print(model.summary())

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_split=0.1)
    # score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # features = features.flatten()


def main():
    wp = WebProcessor(query=['education', 'university', 'college'])
    wp.crawl_website(url='https://illinois.edu/')
    # links = wp.get_hyperlinks()
    tag_text = wp.extract_by_tags()
    tf = wp.get_tfidf(tag_text)
    print(tf)


if __name__ == '__main__':
    # main()
    train_cnn('artificial-intelligence.txt',
              ['education', 'university', 'college'])
