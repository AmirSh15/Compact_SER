import networkx as nx
import numpy as np
import random
import torch
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
import torch.nn.functional as F

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0


def load_data(dataset, Normalize):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    # adj = np.load('dataset/%s/adj.npy' % (dataset))

    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        # n_g = 50
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                    # attr = attr.reshape((68, 2))
                    # attr -= attr[33]
                    # attr[:, 0] /= np.std(attr[:, 0])
                    # attr[:, 1] /= np.std(attr[:, 1])
                    # attr = attr.reshape((68*2))
                    g.add_node(j, att=attr)
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                # if tmp > len(row):
                node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            if not (dataset=="Mine_Graph")or not (dataset == "Mine_Graph_test"):
                assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    #add labels and edge_mat       
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)


    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(g.g.node[0]['att']))
        for i in range(len(g.node_tags)):
                g.node_features[i] = torch.FloatTensor(g.g.node[i]['att'])
                
    ### Normalizing
    if(Normalize):
        X_concat = np.concatenate([graph.node_features.view(-1, graph.node_features.shape[1]) for graph in g_list])
        Min = torch.Tensor(np.min(X_concat, axis=0))[:-2]
        Ptp = torch.Tensor(np.ptp(X_concat, axis=0))[:-2]
        for g in g_list:
            g.node_features[:,:-2] = 2.*(g.node_features[:,:-2] - Min)/Ptp-1
            
    for g in g_list:
        g.node_features2 = torch.zeros(len(g.node_tags), 2*len(g.g.node[0]['att']))
        for i in range(len(g.node_tags)):
            if(i == 0):
                g.node_features2[i] = torch.cat([g.node_features[i], g.node_features[i]])
            else:
                g.node_features2[i] = torch.cat([g.node_features[i], g.node_features[i] - g.node_features[i-1]])
                
    # #### ADJ
    # for g in g_list:
    #     g.adj = torch.zeros(len(g.node_tags), len(g.node_tags))
    #     dist = nn.CosineSimilarity(dim=0, eps=1e-6)
    #     a = []
    #     for i in range(len(g.node_tags)):
    #         for j in range(len(g.node_tags)):
    #             a.append(dist(g.node_features2[i], g.node_features2[j]))
    #     g.adj = F.softmax(torch.Tensor(a), dim=0).view([len(g.node_tags), len(g.node_tags)])



    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)

def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=fold_idx, shuffle = True, random_state = seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
        
    train_portions = []
    test_portions = []
    for j in range(fold_idx):
        train_idx, test_idx = idx_list[j]
    
        train_portions.append([graph_list[i] for i in train_idx])
        test_portions.append([graph_list[i] for i in test_idx])

    return train_portions, test_portions


