class Node:
    weight = None
    def __init__(self, url, state = None, action = None):
        self.url = url
        self.state = state
        self.action = action
        self.Q = self.cal_Q()

    def cal_Q(self):
        #zhangyuan
        pass


#dingcheng aotianyu
class Page:
    def __init__(self, link):
        self.link = link
        self.relevance = None


#huangshuochi
class State:
    def __init__(self, *features):
        pass


#huangshuochi
class Action:
    def __init__(self, *features):
        pass


class IntelliCrawler:
    def __init__(self, url, topic):
        pass
