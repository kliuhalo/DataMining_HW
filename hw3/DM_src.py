import numpy as np
from itertools import permutations
import time

class Graph:
    def __init__(self):
        self.nodes = []
        self.num = 0

    def contains(self, name):
        for node in self.nodes:
            if(node.name == name):
                return True
        return False

    # Return the node with the name, create and return new node if not found
    def find(self, name):
        if(not self.contains(name)):
            new_node = Node(name)
            self.nodes.append(new_node)
            return new_node
        else:
            return next(node for node in self.nodes if node.name == name)
    
    def add_edge(self, parent, child):
        parent_node = self.find(parent)
        child_node = self.find(child)

        parent_node.link_child(child_node)
        child_node.link_parent(parent_node)

    def sort_nodes(self):
        self.nodes.sort(key = lambda x : int(x.name))
    
    def normalize_auth_hub(self):
        auth_sum = sum(node.auth for node in self.nodes)
        hub_sum = sum(node.hub for node in self.nodes)

        for node in self.nodes:
            node.auth /= auth_sum
            node.hub /= hub_sum

    def display_hub_auth(self):
        lst_auth = [node.auth for node in self.nodes]
        lst_hub = [node.hub for node in self.nodes]
        np.savetxt('graph_'+ str(self.num)+'_HITS_authority'  + '.txt',np.asarray(lst_auth),fmt='% 1.5f',delimiter=' ',newline='' )
        np.savetxt('graph_'+str(self.num)+'_HITS_hub_' + '.txt',np.asarray(lst_hub),fmt='% 1.5f',delimiter=' ',newline = '')
        # for node in self.nodes:
        #     #print(f'{node.name}  Auth: {node.old_auth}  Hub: {node.old_hub}')
        #     print(f'{node.name} Auth:{node.auth} Hub:{node.hub}')
        print('Authority:')
        print(lst_auth)
        print('Hub:')
        print(lst_hub)
        print()
        

    def normalize_pagerank(self):
        pagerank_sum = sum(node.pagerank for node in self.nodes)

        for node in self.nodes:
            node.pagerank /= pagerank_sum
    
    def get_pagerank_list(self):
        pagerank_list = np.asarray([node.pagerank for node in self.nodes], dtype='float32')
        #print(pagerank_list)
        return np.round(pagerank_list, 3) 

    def display_pagerank_list(self):
        lst = [node.pagerank for node in self.nodes]
        np.savetxt('graph_'+ str(self.num)+'_PageRank' + str(self.num) + '.txt',np.asarray(lst),fmt='% 1.5f',delimiter=' ',newline = '')
        # for node in self.nodes:
        #     print(f'{node.name} {node.pagerank}')
        print('PageRank:')
        print(lst)
        print()
    
class Node:
    # constructure
    def __init__(self, name):
        self.name = name
        self.children = []
        self.parents = []
        self.auth = 1.0
        self.hub = 1.0
        # for epsilon cal
        # self.old_auth = 0.0
        # self.old_hub = 0.0
        # self.eps = 100

        self.pagerank = 1.0

    # method (HITS)
    def HITS_one_iter(graph):
        node_list = graph.nodes

        for node in node_list:
            node.update_auth()
        for node in node_list:
            node.update_hub()

        graph.normalize_auth_hub()

    def link_child(self, new_child):
        for child in self.children:
            if(child.name == new_child.name):
                return None
        self.children.append(new_child)

    def link_parent(self, new_parent):
        for parent in self.parents:
            if(parent.name == new_parent.name):
                return None
        self.parents.append(new_parent)
    
    #########
    def update_auth(self):
        self.old_auth = self.auth
        self.auth = sum(node.hub for node in self.parents)
        #self.eps += abs(self.auth - self.old_auth)
        

    def update_hub(self):
        self.old_hub = self.hub
        self.hub = sum(node.auth for node in self.children)
        #self.eps += abs(self.hub - self.old_auth)

    ########
    def normalize_pagerank(self):
        pagerank_sum = sum(node.pagerank for node in self.nodes)

        for node in self.nodes:
            node.pagerank /= pagerank_sum

    def update_pagerank(self, d, n):
    #def update_pagerank(self, d):
        in_nodes = self.parents
        # sum pagerank(ni) / C(ni)
        pagerank_sum = sum((node.pagerank / len(node.children)) for node in in_nodes)
        random_jumping = d / n
        self.pagerank = random_jumping + (1-d) * pagerank_sum
        #self.pagerank = d + (1-d) * pagerank_sum


def make_graph(lines):
    graph = Graph()
    for line in lines:
        [parent, child] = line.strip().split(',')
        #print([int(parent), int(child)])
        graph.add_edge(parent, child)
    graph.sort_nodes 
    return graph

def make_graph_ibm(ibm_dict):
    graph = Graph()
    #print(len(ibm_dict))
    lst = []
    for itemset in ibm_dict.items():
        #print(list(permutations(itemset[1],2)))
        lst.append(list(permutations(itemset[1],2)))
    #print(len(lst))
    for i in range(len(lst)):
        for j in lst[i]:
            
            [parent, child] = list(j)
            #print([parent,child])
            graph.add_edge(parent,child)
        graph.sort_nodes 
    return graph  
    

    #     [parent, child] = line.strip().split(',')
    # for   
    #     graph.add_edge(parent, child)
    # graph.sort_nodes 
    # return graph

class Similarity:

    def __init__(self, graph, decay_factor):
        self.decay_factor = decay_factor
        self.name_list, self.old_sim = self.init_sim(graph)
        # name_list 是所有nodeㄉ名字 ，old_sim 為一開始initialize
        self.node_num = len(self.name_list)
        self.new_sim = [[0] * self.node_num for i in range(self.node_num)]
        self.num = graph.num

    # initialize similarity 
    def init_sim(self, graph):
        nodes = graph.nodes
        name_list = [node.name for node in nodes]
        sim = []
        for name1 in name_list:
            temp_sim = []
            for name2 in name_list: #(ex node1 -> temp_sim.append[1,0,0,0,0,0])
                if(name1 == name2):
                    temp_sim.append(1)
                else:
                    temp_sim.append(0)
            sim.append(temp_sim) # sim is a 2d-array of similarity
        
        return name_list, sim

    def get_name_index(self, name): # return the index of node X in node array(name_list)
        return (self.name_list.index(name))

    def get_sim_value(self, node1, node2): # return S(a,b)
        node1_idx = self.get_name_index(node1.name)
        node2_idx = self.get_name_index(node2.name)
        return self.old_sim[node1_idx][node2_idx]
    
    def update_sim_value(self, node1, node2, value): # update to new similarity
        node1_idx = self.get_name_index(node1.name)
        node2_idx = self.get_name_index(node2.name)
        self.new_sim[node1_idx][node2_idx] = value
    
    def replace_sim(self):
        for i in range(len(self.new_sim)):
            self.old_sim[i] = self.new_sim[i]
    
    def calculate_SimRank(self, node1, node2):
        # Return 1 if it's same node
        if(node1.name == node2.name):
            return 1.0

        in_neighbors1 = node1.parents
        in_neighbors2 = node2.parents
        # Return 0 if one of them has no in-neighbor
        if(len(in_neighbors1) == 0 or len(in_neighbors2) == 0):
            return 0.0

        SimRank_sum = 0
        for in1 in in_neighbors1:
            for in2 in in_neighbors2:
                SimRank_sum += self.get_sim_value(in1, in2)

        # Follows the equation S(a,b) = (C / |I(a)||I(b)|) * sum of S(Ii(a),Ii(b))
        scale = self.decay_factor / (len(in_neighbors1) * len(in_neighbors2))
        new_SimRank = scale * SimRank_sum

        return new_SimRank

