import argparse

from focused_crawler import Focused_Crawler_Reinforcement_Learning
from ground_truth import Ground_Truth

def Main(input_all, W2V, collect):
    parser = argparse.ArgumentParser(description='Focused Crawler')
    # input
    parser.add_argument('--seeds', default=['https://cs.illinois.edu/'], help='the start \
                        domains for crawling')
    parser.add_argument('--limit_pages', default=30, help='the maximum \
                        websites to visit')
    parser.add_argument('--topics', default='artificial intelligence', help='target \
                        topics')
    # Q learning parameters
    parser.add_argument('--epsilon', default=0.1, help='epsilon-greedy policy')
    parser.add_argument('--alpha', default=0.01, help='learning rate')
    parser.add_argument('--gamma', default=0.9, help='discount rate')
    # change of relevance
    parser.add_argument('--beta', default=0.5, help='relevance change of target topics')
    parser.add_argument('--sigma1', default=0.1, help='first difference parameter')
    parser.add_argument('--sigma2', default=0.3, help='second difference parameter')
    parser.add_argument('--level', default=3, help='recursively update child for some \
                        levels, may update a page twice')
    # relevant threshold
    parser.add_argument('--relevant', default=0.4, help='the page whose relevance \
                        higher than this value is regarded as relevant page')
    # reward
    parser.add_argument('--reward_true', default=30, help='the reward when page \
                        is relevant')
    parser.add_argument('--reward_false', default=-1, help='the reward when page \
                        is not relevant')
    # mode
    parser.add_argument('--synchronization', default=0, help='the way to \
                        update the value in queue. mode 0, 1, 2 represent: \
                        synchronous, asynchronous, moderated')

    args = parser.parse_args([])
    args.topics = input_all[0]
    args.seeds = input_all[1]
    args.limit_pages = input_all[2]
    Focused_Crawler = Focused_Crawler_Reinforcement_Learning(args.topics, \
        W2V, collect)
    Focused_Crawler.train(args)

    # key_word = 'artificial intelligence'
    # ground_truth = Ground_Truth(key_word, 1000)
    # Focused_Crawler.test(ground_truth)

if __name__ == '__main__':
    main()
