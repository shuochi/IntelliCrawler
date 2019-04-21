import numpy as np

class Web_Processer:

    def __init__(self, topics):
        self.topics = topics
        # self.fake_data_1()
        self.fake_data_2()

    def fake_data_1(self):
        self.links = list(range(1, 12))
        self.outlinks = {1:[2,3,4], 2:[5,6,7], 7:[8,9,10], 9:[11], 10:[4], 11:[1,10]}

    def fake_data_2(self):
        with open('moreno_hens/out.moreno_hens_hens', 'r') as f:
            relation = f.read().split('\n')
        edge = np.zeros((496, 2), dtype=int)
        for i in range(2, len(relation)-1):
            string = relation[i].strip().split(' ')
            edge[i-2] = [int(j) for j in string]
        self.links = list(range(1, 33))
        self.outlinks = {}
        for i in edge:
            if i[0] not in self.outlinks.keys():
                self.outlinks[i[0]] = [i[1]]
            else:
                self.outlinks[i[0]].append(i[1])

    def crawl_website(self, link):
        self.link = link
        # use self.link
        # get self.page

    def page_target_topics(self):
        # use self.page, self.topics
        # get relevance
        relevance = np.random.rand()
        return relevance

    def outlink_target_topics(self, visited):
        # use self.outlinks, self.topics
        # get list_of_relevance
        dict_of_relevance = {}
        if self.link in self.outlinks.keys():
            for i in self.outlinks[self.link]:
                dict_of_relevance[i] = np.random.rand()
        return dict_of_relevance # dict = {outlink: relevance}

if __name__ == '__main__':
    w = Web_Processer('dd')
    print(w.outlinks)
