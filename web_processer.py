import os
import re
import string
from collections import defaultdict

import numpy as np
import requests
from bs4 import BeautifulSoup

import gensim


class WebProcesser:
    def __init__(self, topics, categories=None, model=None, sim_threshold=0.4):
        self.topics = topics
        self.categories = categories
        if model:
            self.model = model
        else:
            path = '../Relevance-HTML/model/GoogleNews-vectors-negative300.bin'
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
        print('Web page loaded..')

    def extract_by_tags(self,
                        tags=[
                            'title', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p',
                            'ul', 'ol', 'table'
                        ]):
        self.tags = tags
        tag_text = defaultdict(str)
        for tag in tags:
            for elem in self.soup.find_all(tag):
                stripped = elem.text.translate(
                    str.maketrans(
                        str.maketrans(dict.fromkeys(string.punctuation))))
                stripped = re.sub(r'\s+', ' ', stripped).strip()
                if stripped:
                    tag_text[tag] += f' {stripped.lower()}'
        return tag_text

    def get_hyperlinks(self):
        self.links = []
        for link in self.soup.find_all('a'):
            self.links.append(link.get('href'))
        return self.links

    def page_categories(self):
        # use self.page, self.categories
        # get relevance
        pass

    def outlink_target_topic(self, index):
        # use self.outlinks, self.topics
        # get list_of_relevance
        for i in index:
            self.outlinks[i]
            pass
        return list_of_relevance

    def outlink_categories(self, index):
        # use self.outlinks, self.categories
        # get list_of_relevance
        for i in index:
            self.outlinks[i]
            pass
        return list_of_relevance

    def calculate_tfidf(self, tag_text):
        """Calculate term frequency (tf) for each tag."""
        tag_tf = {}  # tag: tf
        tag_relevant = {}  # tag: boolean
        total_words = 0  # total # of words in web page
        total_relevant_words = 0  # total # of relevant words in web page
        for tag in self.tags:
            relevant_words = 0
            words = tag_text[tag].split()
            total_words += len(words)
            for word in words:
                try:
                    sim = self.model.similarity(self.topics, word)
                except KeyError:
                    sim = 0
                if sim >= self.sim_threshold:
                    relevant_words += 1
            total_relevant_words += relevant_words
            try:
                tf = relevant_words / len(words)
            except ZeroDivisionError:
                tf = 0
            tag_tf[tag] = tf
            # update idf
            if relevant_words:
                tag_relevant[tag] = True
            else:
                tag_relevant[tag] = False
        idf = np.log10(len(self.tags) / (sum(tag_relevant.values()) + 1))
        tag_tfidf = {}
        for tag in self.tags:
            tag_tfidf[tag] = tag_tf[tag] * idf

        relevance = total_relevant_words / total_words

        return tag_tf, tag_tfidf, relevance


def main():
    wp = WebProcesser(topics='education')
    wp.crawl_website(url='https://illinois.edu/')
    tag_text = wp.extract_by_tags()
    tag_tf, tag_tfidf, relevance = wp.calculate_tfidf(tag_text)
    print(tag_tf, tag_tfidf)
    print(relevance)


if __name__ == '__main__':
    main()
