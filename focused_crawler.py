import numpy as np
import heapq
import networkx as nx

from web_processer import Web_Processer

class Focused_Crawler_Reinforcement_Learning:

    def __init__(self, ground_truth):
        self.ground_truth = ground_truth
        pass

    def train(self, args):
        self.args = args
        self.processer = Web_Processer(args.topic, args.categories)
        self.w = np.zeros(5*(5+len(args.categories)))
        self.B = []
        self.visited = set()
        self.DG = nx.DiGraph()
        visited_pages = 0

        for link in args.seeds:
            state = self.page_state(link, None)
            outlinks = self.processer.outlinks()
            action_index = action_index(outlinks)
            action = self.outlink_action(link, action_index)
            sa_list = self.encode(state, action)
            for i, j in enumerate(action_index):
                heapq.heappush(self.B, [0, link, sa_list[i], outlinks[j]])

        while visited_pages < args.limit_pages:
            if np.random.rand() < args.epsilon:
                pair = self.B.pop(np.random.randint(len(self.B)))
            else:
                pair = heapq.heappop(self.B)
            parent_link, sa, link = pair[1:]
            if link in self.visited:
                continue

            reward = self.ground_truth.reward(link)
            self.processer.crawl_website(link)
            state = self.page_state(link, parent_link)
            outlinks = self.processer.outlinks()
            action_index = action_index(outlinks)
            action = self.outlink_action(link, action_index)
            sa_list = self.encode(state, action)
            Q_list = self.decode(sa_list) @ self.w

            if self.ground_truth.is_relevant(link):
                self.w += args.alpha*(reward-np.inner(self.decode(sa),self.w))*\
                          self.decode(sa)
            else:
                if np.random.rand() < args.epsilon:
                    sa_next = sa_list[np.random.randint(len(sa_list))]
                else:
                    sa_next = sa_list[np.argmax(Q_list)]
                if args.synchronization == 1:
                    self.w += args.alpha*(reward+args.gamma*np.inner(self.decode(\
                              sa_next),self.w))-np.inner(self.decode(sa),self.w))*\
                              self.decode(sa)
                else:
                    self.w += args.alpha*(1-args.gamma)*(reward+args.gamma*\
                              np.inner(self.decode(sa_next),self.w))-np.inner(\
                              self.decode(sa),self.w))*sa.decode(sa)

            if args.synchronization == 0:
                for pair in self.B:
                    pair[0] = np.inner(self.decode(pair[2]), self.w)
                heapq.heapify(self.B)
            for i, j in enumerate(action_index):
                heapq.heappush(self.B, [Q_list[i], link, sa_list[i], outlinks[j]])

            visited_pages += 1

    def page_state(self, link, parent_link):
        self.visited.add(link)
        self.processer.crawl_website(link)
        page_relevant = self.ground_truth.is_relevant(link)
        page_target_topic = self.processer.page_target_topic()
        page_categories = self.processer.page_categories()
        self.DG.add_node(link, relevant=page_relevant, relevance=\
                         page_target_topic, my=0 ,max=0)
        page_change, page_all_parents, page_relevant_parents, page_distance = \
            self.update_state_graph(link, parent_link)
        state = [page_target_topic, page_change, page_categories, \
                 page_all_parents, page_relevant_parents, page_distance]
        return state

    def update_state_graph(self, link, parent_link):
        page_all_parents = 0
        page_relevant_parents = 0
        page_distance = 0 if DGs.nodes[link]['relevant'] else 1
        if not parent_link:
            page_change = 0
        else:
            DG.add_edge(parent_link, link)
            parents = nx.algorithms.dag.ancestors(self.DG, link)
            parents.add(link)
            DGs = self.DG.subgraph(parents)
            parents.remove(link)
            relevant = {}
            for i in parents:
                page_all_parents += DGs.nodes[i]['relevance']
                if DGs.nodes[i]['relevant']:
                    relevant[i] = nx.algorithms.generic.shortest_path_length(DGs,i,link)
                    page_relevant_parents += DGs.nodes[i]['relevance']
            page_all_parents /= len(parents)
            page_relevant_parents /= len(relevant)
            page_distance = min(min(relevant.values())/10, page_distance)
            DGs.nodes[link]['max'] = max(DGs.nodes[link]['max'], \
                                         DGs.nodes[parent_link]['my'])
            DGs.nodes[link]['my'] = self.args.beta*DGs.nodes[link]['relevance']\
                +(1-self.args.beta)*DGs.nodes[link]['max']
            page_change = DGs.nodes[link]['relevance'] - DGs.nodes[link]['max']
            self.recursive_update_child(link)
        return page_change, page_all_parents, page_relevant_parents, page_distance

    def recursive_update_child(self, parent):
        if DGs.out_degree(parent):
            for i in DGs.successors(parent):
                DGs.nodes[i]['max'] = max(DGs.nodes[i]['max'], \
                                             DGs.nodes[parent]['my'])
                DGs.nodes[i]['my'] = self.args.beta*DGs.nodes[i]['relevance']\
                    +(1-self.args.beta)*DGs.nodes[i]['max']
                self.recursive_update_child(i)

    def action_index(self, outlinks):
        action_index = []
        for i, link in enumerate(outlinks):
            if link not in self.visited:
                action_index.append(i)
        return action_index

    def outlink_action(self, link, action_index):
        outlink_target_topic = self.processer.outlink_target_topic(action_index)
        outlink_categories = self.processer.outlink_categories(action_index)
        outlink_all_parents = 0
        outlink_relevant_parents = 0
        parents = nx.algorithms.dag.ancestors(self.DG, link)
        parents.add(link)
        DGs = self.DG.subgraph(parents)
        relevant = {}
        for i in parents:
            outlink_all_parents += DGs.nodes[i]['relevance']
            if DGs.nodes[i]['relevant']:
                relevant[i] = nx.algorithms.generic.shortest_path_length(DGs,i,link)
                outlink_relevant_parents += DGs.nodes[i]['relevance']
        outlink_all_parents /= len(parents)
        outlink_relevant_parents /= len(relevant)
        action = [outlink_target_topic, outlink_categories, \
                  outlink_all_parents, outlink_relevant_parents]
        return action

    def encode(self, state, action):
        sa_list = []
        for action_i in range(len(action)):
            sa = ''
            for i in range(6):
                if i == 1:
                    if -self.args.sigma1 <= state[i] <= self.args.sigma1:
                        sa +='0'
                    elif self.args.sigma1 < state[i] <= self.args.sigma2:
                        sa +='1'
                    elif self.args.sigma2 < state[i] <= 1:
                        sa +='2'
                    elif -self.args.sigma2 <= state[i] < -self.args.sigma1:
                        sa +='3'
                    elif -1 <= state[i] < -self.args.sigma2:
                        sa +='4'
                elif i == 2:
                    for j in state[i]:
                        index = int(j*10//2)
                        sa += str(4 if index == 5 else index)
                else:
                    index = int(state[i])*10//2)
                    sa += str(4 if index == 5 else index)
            for i in range(4):
                if i == 1:
                    for j in action[action_i][i]:
                        index = int(j*10//2)
                        sa += str(4 if index == 5 else index)
                else:
                    index = int(action[action_i])*10//2)
                    sa += str(4 if index == 5 else index)
            sa_list.append(sa)
        return sa_list

    def decode(self, sa):
        if type(sa) == str:
            q = np.zeros(5*len(sa))
            for i in range(len(sa)):
                q[int(sa[i])+i*5] = 1
        else:
            q = np.zeros((len(sa), 5*len(sa[0])))
            for i in range(len(sa)):
                for j in range(len(sa[0])):
                    q[i, int(sa[i][j])+j*5] = 1
        return q
