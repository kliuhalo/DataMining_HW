import os
import sys
from DM_src import *
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import pandas as pd
import time

def HITS_one_iter(graph):
    node_list = graph.nodes

    for node in node_list:
        node.update_auth()

    for node in node_list:
        node.update_hub()

    graph.normalize_auth_hub()
    
def HITS(graph, iter):
    for i in range(iter):
        HITS_one_iter(graph)
    graph.display_hub_auth()
    return graph

def PageRank(graph,damping_factor,iter):
    for i in range(iter):
        PageRank_one_iter(graph, damping_factor)
    graph.display_pagerank_list()
    

def PageRank_one_iter(graph, d):
    node_list = graph.nodes
    for node in node_list:
        node.update_pagerank(d, len(graph.nodes))
    graph.normalize_pagerank()
    #print(graph.get_pagerank_list())

def SimRank_one_iter(graph, sim):
    for node1 in graph.nodes:
        for node2 in graph.nodes:
            new_SimRank = sim.calculate_SimRank(node1, node2)
            sim.update_sim_value(node1, node2, new_SimRank)
    #print(sim.new_sim)
    sim.replace_sim()

def SimRank(graph, sim, iteration=100):
    for i in range(iteration):
        SimRank_one_iter(graph, sim)
   
def pre_ibm():
    path = os.path.join(os.path.dirname(__file__),'hw3dataset/ibm-2021.txt')
    database = open(path,'r')
    list_dict = {}
    list_dict = defaultdict(list)
    for i in database:
        list_dict[int(i.split()[1])].append(int(i.split()[2]))
    #print(list_dict)
    return list_dict

if __name__ == "__main__":
    if str(sys.argv[1]) == "graph_1":
        with open('hw3dataset/graph_1.txt','r') as f:
            lines = f.readlines()
            graph = make_graph(lines)
            graph.num = 1

    elif str(sys.argv[1]) == "graph_2":
        with open('hw3dataset/graph_2.txt','r') as f: 
            lines = f.readlines()
            graph = make_graph(lines)
            graph.num = 2

    elif str(sys.argv[1]) == "graph_3":
        with open('hw3dataset/graph_3.txt','r') as f:
            lines = f.readlines()
            graph = make_graph(lines)
            graph.num = 3

    elif str(sys.argv[1]) == "graph_4":
        with open('hw3dataset/graph_4.txt','r') as f:
            lines = f.readlines()
            graph = make_graph(lines)
            graph.num = 4

    elif str(sys.argv[1]) == "graph_5":
        with open('hw3dataset/graph_5.txt','r') as f:
            lines = f.readlines()
            graph = make_graph(lines)
            graph.num = 5

    elif str(sys.argv[1]) == "graph_6":
        with open('hw3dataset/graph_6.txt','r') as f:
            lines = f.readlines()
            graph = make_graph(lines)
            graph.num = 6
    else:
        list_dict = pre_ibm()
        print(list_dict)
        graph = make_graph_ibm(list_dict)
        graph.num = 7

    iter = 100
    start = time.time()
    HITS(graph,iter)
    print("Total Time of HITS: ", time.time()-start, " sec")
    #print(list_dict)
    damping_factor = 0.1
    start = time.time()
    PageRank(graph, damping_factor, iter)
    print("Total Time of pagerank: ", time.time()-start, " sec")
    
    
    decay_factor = 0.9
    iter = 10
    sim = Similarity(graph,decay_factor)
    start = time.time()
    SimRank(graph,sim,iter)
    print("Total Time of SimRank: ", time.time()-start, " sec")
    #print(sim.new_sim, sim.old_sim)
    print('SimRank: ')
    print(np.asarray(sim.new_sim))
    print()
    np.savetxt('graph_'+str(graph.num)+'_SimRank.txt',np.asarray(sim.new_sim),fmt='% 1.5f')
    
