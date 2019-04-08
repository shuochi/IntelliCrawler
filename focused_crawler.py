import random

from web_processer import Web_Processer

class Focused_Crawler_Reinforcement_Learning:

    def __init__(self, ground_truth):
        self.ground_truth = ground_truth
        pass

    def page_state(self):
        page_target_topic = self.processer.page_target_topic()
        page_categories = self.processer.page_categories()
        pass

    def outlink_actions(self, actions_index):
        outlink_target_topic = self.processer.outlink_target_topic(actions_index)
        outlink_categories = self.processer.outlink_categories(actions_index)
        pass

    def B_enqueue(self, outlinks, state, actions, Q):
        pass

    def actions_index(self, outlinks):
        return actions_index

    def train(self, args):
        self.processer = Web_Processer(args.topics, args.categories)
        w = initialize_w()
        B = priority_queue_B()
        self.visited = set()
        visited_pages = 0

        for link in args.seeds:
            self.processer.crawl_website(link)
            state = self.page_state()
            outlinks = self.processer.outlinks()
            actions_index = range(len(outlinks))
            actions = self.outlink_actions(actions_index)
            Q = 0
            self.B_enqueue(outlinks, state, actions, Q)

        while visited_pages < args.limit_pages:
            if random.random() < self.epsilon:
                l_s_a = select_random()
            else:
                l_s_a = select_max_Q()
            link = l_s_a[0]
            if link in self.visited:
                continue

            reward = self.ground_truth.reward(link)
            self.processer.crawl_website(link)
            state = self.page_state()
            outlinks = self.processer.outlinks()
            actions_index = actions_index(outlinks)
            actions = self.outlink_actions(actions_index)

            if is_relevant(l_s_a[0]):
                pass
            else:
                pass

            if args.synchronization == 0:
                for pair in B:
                    pass
            else:
                for pair in B:
                    pass

            visited_pages += 1
