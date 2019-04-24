import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import nltk
from nltk.corpus import wordnet
from web_processor import WebProcessor
import gensim

# Build CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=10, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=1)

    def forward(self, input):
        output = self.fc1(input)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)

        return output

class LinkEvaluation:
    def __init__(self, topic_list, processor):
        self.topic_list = topic_list
        self.processor = processor
        self.net = torch.load('url_relevance.pth')

    def get_feature(self, links, topic, relevant_urls):

        # Calculate the edit distance between candidate url and relevant url
        def edit_distance(word1, word2):
            l1, l2 = len(word1)+1, len(word2)+1
            dp = [[0 for _ in range(l2)] for _ in range(l1)]
            for i in range(l1):
                dp[i][0] = i
            for j in range(l2):
                dp[0][j] = j
            for i in range(1, l1):
                for j in range(1, l2):
                    dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(word1[i-1]!=word2[j-1]))
            return dp[-1][-1]

        # Parse url to n-gram
        def parse_gram(url, n):
            res = set()
            for i in range(len(url) - n + 1):
                res.add(url[i:i + n])
            return res

        url_feature = {}

        # Preprocess to remove http/https/www/non-letter
        for i in range(len(relevant_urls)):
            temp = relevant_urls[i][0].replace("http://","").replace("https://","").replace("www.","")
            relevant_urls[i] = (''.join(filter(str.isalpha, temp)), relevant_urls[i][1])

        for _url, _text in links.items():
            # Preprocess url to remove to remove http/https/www/non-letter
            url = ''.join(filter(str.isalpha, _url.replace("http://","").replace("https://","").replace("www.","")))
            text = _text.split(' ')

            # Edit Distance, tf, 2,3,4,5,6,7-gram apperance rate, Word similarity, Word synonym
            feature = np.zeros(10)

            if len(relevant_urls) > 0:
                # Average edit distance of all relevent url
                for relevant_url, relevant_score in relevant_urls:
                    feature[0] += (len(url) - edit_distance(url, relevant_url)) * relevant_score
                feature[0] = feature[0] / len(relevant_urls)

                # 2,3,4,5,6,7-gram apperance rate
                for ngram in range(2, 8):
                    url_parse = parse_gram(url, ngram)
                    for relevant_url, relevant_score in relevant_urls:
                        relevant_url_parse = parse_gram(relevant_url, ngram)
                        feature[ngram] += len(url_parse & relevant_url_parse) * relevant_score
                    feature[ngram] = feature[ngram] / len(relevant_urls)


            for word in text:
                # Tf
                if word in topic or topic in word:
                     feature[1] += 1

                # Word similarity and synonym
                try:
                    feature[-2] += self.processor.model.similarity(topic, word)
                except Exception:
                    feature[-2] += 0
                try:
                    w1 = wordnet.synset(topic + '.n.01')
                    w2 = wordnet.synset(word + '.n.01')
                    feature[-1] += w1.wup_similarity(w2)
                except Exception:
                    feature[-1] += 0

            feature[1] = feature[1] / len(text)
            feature[-2] = feature[-2] / len(text)
            feature[-1] = feature[-1] / len(text)

            url_feature[_url] = feature

        return url_feature

    def get_link_score(self, relevant_urls):
        links = self.processor.get_hyperlinks()
        res_score = np.zeros(len(links))
        for topic in self.topic_list:
            features = self.get_feature(links, topic, relevant_urls)
            features_data = [(torch.FloatTensor(list(features.values())[i])) for i in range(0, len(list(features.values())))]
            features_loader = DataLoader(dataset=features_data, batch_size=1, shuffle=False)
            features_score = []
            for data in features_loader:
                features_score.append(self.net(data).item())
            res_score += np.array(features_score)
        res_score /= len(self.topic_list)

        res = {}
        index = 0
        for key in links.keys():
            res[key] = res_score[index]
            index += 1

        return res

if __name__ == '__main__':
    L = LinkEvaluation('sdf', 'sdf')
