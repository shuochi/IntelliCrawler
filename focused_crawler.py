import numpy as np
import heapq
import networkx as nx
import matplotlib.pyplot as plt

# from fake_web import FakeWeb
from web import Web

class Focused_Crawler_Reinforcement_Learning:
    def __init__(self, args):
        self.args = args
        self.processer = Web(self.args.topics)

    def train(self):
        self.w = np.zeros(5*(5+3))
        self.B = []
        self.visited = set()
        self.relevant = []
        self.DG = nx.DiGraph()
        visited_pages = 0

        for link in self.args.seeds:
            _, state = self.page_state(link, None) # 5 relevance
            # list_outlinks (unvisited), 1 list_relevance + 2 relevance
            outlinks, action = self.outlink_action(link)
            sas = self.encode(state, action) # list_code (8 digits)
            for i in range(len(outlinks)):
                heapq.heappush(self.B, [0, link, outlinks[i], sas[i]])
            self.log(link, None, outlinks, None)

        while visited_pages < self.args.limit_pages:
            if not len(self.B):
                break
            if np.random.rand() < self.args.epsilon:
                pair = self.B.pop(np.random.randint(len(self.B)))
            else:
                pair = heapq.heappop(self.B)
            parent_link, link, sa = pair[1:]
            if link in self.visited:
                self.DG.add_edge(parent_link, link)
                self.recursive_update(parent_link, link)
                print('have visited!')
                self.log(link, parent_link, None, None)
                continue

            reward, state = self.page_state(link, parent_link)
            outlinks, action = self.outlink_action(link)
            if len(outlinks):
                sas = self.encode(state, action)
                Q_list = self.decode(sas) @ self.w

            q_this = np.inner(self.decode(sa),self.w)
            if reward == self.args.reward_true:
                self.w += self.args.alpha*(reward-q_this)*self.decode(sa)
            else:
                q_next = 0
                if len(outlinks):
                    if np.random.rand() < self.args.epsilon:
                        sa_next = sas[np.random.randint(len(sas))]
                    else:
                        sa_next = sas[np.argmax(Q_list)]
                    q_next = np.inner(self.decode(sa_next),self.w)
                if self.args.synchronization == 2:
                    self.w += self.args.alpha*(1-self.args.gamma)*(reward+\
                              self.args.gamma*q_next-q_this)*self.decode(sa)
                else:
                    self.w += self.args.alpha*(reward+self.args.gamma*q_next-\
                              q_this)*self.decode(sa)

            if self.args.synchronization == 0:
                for pair in self.B:
                    pair[0] = -np.inner(self.decode(pair[3]), self.w)
                heapq.heapify(self.B)
            for i in range(len(outlinks)):
                heapq.heappush(self.B, [-Q_list[i], link, outlinks[i], sas[i]])

            self.log(link, parent_link, outlinks, reward)
            visited_pages += 1

        print('Final w:', self.w)
        f = plt.figure()
        nx.draw_circular(self.DG, with_labels=True, ax=f.add_subplot(111))
        f.savefig('graph.png')

    def test(self, ground_truth):
        pass

    def page_state(self, link, parent_link):
        page_target_topics = self.processer.page_target_topics(link)
        page_relevant = True if page_target_topics >= self.args.relevant else False
        if page_relevant:
            self.relevant.append((link, page_target_topics))
        reward = self.args.reward_true if page_relevant else self.args.reward_false
        self.visited.add(link)
        self.DG.add_node(link, relevant=page_relevant, relevance=\
                         page_target_topics, my=page_target_topics ,max=0)
        page_change, page_all_parents, page_relevant_parents, page_distance = \
            self.update_state_graph(link, parent_link)
        state = [page_target_topics, page_change, page_all_parents, \
                 page_relevant_parents, page_distance]
        return reward, state

    def update_state_graph(self, link, parent_link):
        page_all_parents = 0
        page_relevant_parents = 0
        page_distance = 0 if self.DG.nodes[link]['relevant'] else 1
        if not parent_link:
            page_change = 0
        else:
            self.DG.add_edge(parent_link, link)
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
            if page_all_parents:
                page_all_parents /= len(parents)
            if page_relevant_parents:
                page_relevant_parents /= len(relevant)
            if relevant.values():
                page_distance = min(min(relevant.values())/10, page_distance)
            self.recursive_update(parent_link, link)
            page_change = self.DG.nodes[link]['relevance'] - self.DG.nodes[link]['max']
        return page_change, page_all_parents, page_relevant_parents, page_distance

    def recursive_update(self, parent_link, link):
        self.DG.nodes[link]['max'] = max(self.DG.nodes[link]['max'], \
                                     self.DG.nodes[parent_link]['my'])
        self.DG.nodes[link]['my'] = self.args.beta*self.DG.nodes[link]['relevance']\
            +(1-self.args.beta)*self.DG.nodes[link]['max']
        self.recursive_update_child(link, self.args.level)

    def recursive_update_child(self, parent, n):
        if self.DG.out_degree(parent) and n:
            for i in self.DG.successors(parent):
                self.DG.nodes[i]['max'] = max(self.DG.nodes[i]['max'], \
                                             self.DG.nodes[parent]['my'])
                self.DG.nodes[i]['my'] = self.args.beta*self.DG.nodes[i]['relevance']\
                    +(1-self.args.beta)*self.DG.nodes[i]['max']
                self.recursive_update_child(i, n-1)

    def outlink_action(self, link):
        outlink_action = self.processer.outlink_target_topics(self.relevant)
        outlinks = []
        outlink_target_topics = []
        for key in outlink_action.keys():
            if key not in self.visited:
                outlinks.append(key)
                outlink_target_topics.append(outlink_action[key])
            else:
                self.DG.add_edge(link, key)
                self.recursive_update(link, key)
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
        if outlink_all_parents:
            outlink_all_parents /= len(parents)
        if outlink_relevant_parents:
            outlink_relevant_parents /= len(relevant)
        action = [outlink_target_topics, outlink_all_parents, outlink_relevant_parents]
        return outlinks, action

    def encode(self, state, action):
        sas_state = ''
        for i in range(5):
            if i == 1:
                if -self.args.sigma1 <= state[i] <= self.args.sigma1:
                    sas_state +='0'
                elif self.args.sigma1 < state[i] <= self.args.sigma2:
                    sas_state +='1'
                elif self.args.sigma2 < state[i] <= 1:
                    sas_state +='2'
                elif -self.args.sigma2 <= state[i] < -self.args.sigma1:
                    sas_state +='3'
                elif -1 <= state[i] < -self.args.sigma2:
                    sas_state +='4'
            else:
                index = int(state[i]*10//2)
                sas_state += str(4 if index == 5 else index)
        sas = []
        for action_i in range(len(action[0])):
            sas_action = ''
            for i in range(3):
                if i == 0:
                    index = int(action[0][action_i]*10//2)
                    sas_action += str(4 if index == 5 else index)
                else:
                    index = int(action[i]*10//2)
                    sas_action += str(4 if index == 5 else index)
            sas.append(sas_state + sas_action)
        return sas

    def decode(self, sa):
        if type(sa) == str:
            q = np.zeros(5*len(sa))
            for i in range(len(sa)):
                q[int(sa[i])+i*5] = 1
        elif type(sa) == list:
            q = np.zeros((len(sa), 5*len(sa[0])))
            for i in range(len(sa)):
                for j in range(len(sa[0])):
                    q[i, int(sa[i][j])+j*5] = 1
        return q

    def log(self, link, parent_link, outlinks, reward):
        print('link:', link)
        if not parent_link:
            print('parent_link: NULL')
        else:
            print('parent_link:', parent_link)
        if not outlinks:
            print('outlinks: NULL')
        else:
            print('outlinks:', outlinks)
        if not reward:
            print('reward: NULL')
        else:
            print('reward:', reward)
        print('DG:', self.DG.nodes.data())
        print('B:', self.B)
        input('-'*80)
        # print('-'*80)
