from azure.cognitiveservices.search.websearch import WebSearchAPI
from azure.cognitiveservices.search.websearch.models import SafeSearch
from msrest.authentication import CognitiveServicesCredentials

subscription_key = "303c93e3970f4bfca81f44a53f5c58f5"
client = WebSearchAPI(CognitiveServicesCredentials(subscription_key))

search = 'chengxiang zhai site:illinois.edu'
web_data = client.web.search(query=search, response_filter=['Webpages'], client_ip='130.126.255.152', count=5)
print('\nSearched for Query# {}'.format(search))

'''
Web pages
If the search response contains web pages, the first result's name and url
are printed.
'''
if hasattr(web_data.web_pages, 'value'):

    print("\nWebpage Results#{}".format(len(web_data.web_pages.value)))

    for page in web_data.web_pages.value:
        print("First web page name: {} ".format(page.name))
        print("First web page URL: {} ".format(page.url))

else:
    print("Didn't find any web pages...")
