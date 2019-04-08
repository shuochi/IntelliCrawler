import argparse

from focused_crawler import Focused_Crawler_Reinforcement_Learning
from ground_truth import Ground_Truth

def main():
    parser = argparse.ArgumentParser(description='Focused Crawler')
    # input
    parser.add_argument('--seeds', default=['illinois.edu'], help='the start \
                        domains for crawling')
    parser.add_argument('--limit_pages', default=1000, help='the maximum \
                        websites to visit')
    parser.add_argument('--topics', default=['chengxiang zhai'], help='target \
                        topics')
    parser.add_argument('--categories', default=[], help='related categories \
                        with topics, get from Open Directory Project')
    # Q learning parameters
    parser.add_argument('--epsilon', default=0.2, help='epsilon-greedy policy')
    parser.add_argument('--alpha', default=0.2, help='learning rate')
    parser.add_argument('--gamma', default=0.2, help='discount rate')
    # mode
    parser.add_argument('--synchronization', default=0, help='the way to \
                        update the value in queue. mode 0, 1, 2 represent: \
                        synchronous, asynchronous, moderated')
    args = parser.parse_args()

    key_word = 'chengxiang zhai site:illinois.edu'
    ground_truth = Ground_Truth(key_word, 200)
    # Focused_Crawler = Focused_Crawler_Reinforcement_Learning(ground_truth)
    # Focused_Crawler.train(args)

if __name__ == '__main__':
    main()
