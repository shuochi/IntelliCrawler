from bing.bing_search import Bing_Search

class Ground_Truth:

    def __init__(self, key_word, num):
        self.key_word = key_word
        self.num = num
        self.bing_result()
        self.illinois_result()
        self.result = set()
        self.result.update(self.bing_url)

    def bing_result(self):
        exist = 1
        try:
            with open('bing_url.txt', 'r') as f:
                self.bing_url = f.read().split('\n')
            with open('bing_name.txt', 'r', encoding='utf-8') as f:
                self.bing_name = f.read().split('\n')
            title = self.key_word + ', ' + str(self.num)
            if self.bing_url.pop(0) != title or self.bing_name.pop(0) != title:
                exist = 0
        except:
            exist = 0
        if exist == 0:
            b = Bing_Search()
            self.bing_url, self.bing_name = b.search(self.key_word, self.num)
        print('Bing result:\nSearch={}\nCount={}'.format(self.key_word, self.num))

    def illinois_result(self):
        pass

    def reward(self, link):
        if link in self.result:
            return 30
        else:
            return -1
