from azure.cognitiveservices.search.websearch import WebSearchAPI
from azure.cognitiveservices.search.websearch.models import SafeSearch
from msrest.authentication import CognitiveServicesCredentials

class Bing_Search:
    def __init__(self):
        subscription_key = "303c93e3970f4bfca81f44a53f5c58f5"
        self.client = WebSearchAPI(CognitiveServicesCredentials(subscription_key))

    def search(self, key_word, num):
        bing_url = []
        bing_name = []
        visited = 0
        while visited < num:
            web_data = self.client.web.search(query=key_word, \
                response_filter=['Webpages'], client_ip='130.126.255.152', \
                count=num-visited, offset=visited)
            visited += len(web_data.web_pages.value)
            for page in web_data.web_pages.value:
                bing_url.append(page.url)
                bing_name.append(page.name)
        title = key_word + ', ' + str(visited)
        with open('bing_url.txt', 'w') as f:
            f.write(title+'\n')
            f.write('\n'.join(bing_url))
        with open('bing_name.txt', 'w', encoding='utf-8') as f:
            f.write(title+'\n')
            f.write('\n'.join(bing_name))
        return bing_url, bing_name

if __name__ == '__main__':

    subscription_key = "303c93e3970f4bfca81f44a53f5c58f5"
    client = WebSearchAPI(CognitiveServicesCredentials(subscription_key))
    search = 'chengxiang zhai site:illinois.edu'
    web_data = client.web.search(query=search, response_filter=['Webpages'], \
        client_ip='130.126.255.152', count=5)
    print('\nSearched for Query# {}'.format(search))

    if hasattr(web_data.web_pages, 'value'):
        print("\nWebpage Results#{}".format(len(web_data.web_pages.value)))
        for index, page in enumerate(web_data.web_pages.value):
            print("{} web page name: {} ".format(index+1, page.name))
            print("{} web page URL: {} ".format(index+1, page.url))
    else:
        print("Didn't find any web pages...")
