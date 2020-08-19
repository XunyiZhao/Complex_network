import numpy as np
import pandas as pd
import networkx as nx
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import argparse

work1 = pd.read_table('datasets/tij_InVS13.dat', sep=" ", header = None, names = ['time','p1','p2'])
work2 = pd.read_table('datasets/tij_InVS15.dat', sep=" ", header = None, names = ['time','p1','p2'])
hs11 = pd.read_csv('datasets/thiers_2011.csv', delimiter='\t',names = ['time','p1','p2','c1','c2'])[['time','p1','p2']]
hs12 = pd.read_csv('datasets/thiers_2012.csv', delimiter=',',names = ['time','p1','p2','c1','c2'])[['time','p1','p2']]

mit = pd.read_table('datasets/out.mit', sep="\s+|\t", skiprows=1, names = ['p1','p2','w','time'],engine='python')
mit = mit[['time','p1','p2']]
start1, start2 = np.random.choice(mit['time'],2)
mit1 = pd.DataFrame.copy(mit[(mit['time']<start1+24*3600*7)&(mit['time']>start1)])
mit2 = pd.DataFrame.copy(mit[(mit['time']<start2+24*3600*7)&(mit['time']>start2)])

datasets = {'hs11':hs11,'hs12':hs12,'work1':work1,'work2':work2,'mit1':mit1,'mit2':mit2}

class model():
    def __init__(self, contacts, modes, block_prob=0.1):
        self.contacts = contacts
        self.modes = modes
        self.people = np.unique(self.contacts[['p1','p2']])
        self.n = len(self.people)
        self.block_prob = {}
        p2id = {}
        for i, p in enumerate(self.people):
            p2id[p] = i
        self.contacts['id1'] = self.contacts['p1'].apply (lambda row: p2id[row])
        self.contacts['id2'] = self.contacts['p2'].apply (lambda row: p2id[row])

        # aggregate the contacts
        self.backbones = np.zeros([self.n,self.n])
        for i, c in self.contacts.iterrows():
            t,p1,p2 = c[['time','id1','id2']]
            self.backbones[p1,p2] += 1
            self.backbones[p2,p1] += 1

        self.agg_graph = nx.from_numpy_array(self.backbones)
        
        for mode in self.modes:
            if mode =='degree product':
                adj = (self.backbones>0)
                degree = np.sum(adj,axis=0)
                feature = np.outer(degree, degree)
            elif mode =='r degree product':
                adj = (self.backbones>0)
                degree = np.sum(adj,axis=0)
                feature = 1/np.outer(degree, degree)
            elif mode =='strength product':
                strength = np.sum(self.backbones,axis=0)
                feature = np.outer(strength, strength)
            elif mode =='r strength product':
                strength = np.sum(self.backbones,axis=0)
                feature = 1/np.outer(strength, strength)
            elif mode =='betweeness':
                feature = np.zeros([self.n,self.n])
                bet = nx.algorithms.edge_betweenness_centrality(self.agg_graph)
                for k,v in bet.items():
                    feature[k] = v
                feature += feature.T
            elif mode =='r betweeness':
                feature = np.zeros([self.n,self.n])
                bet = nx.algorithms.edge_betweenness_centrality(self.agg_graph)
                for k,v in bet.items():
                    if v>0:
                        feature[k] = 1/v
                feature += feature.T
            elif mode =='link weight':
                feature = np.copy(self.backbones)
            elif mode =='r link weight':
                feature = np.zeros([self.n,self.n])
                feature[self.backbones!=0] = 1/self.backbones[self.backbones!=0]
            elif mode =='weighted eigen':
                eigen = nx.eigenvector_centrality_numpy(self.agg_graph, weight = 'weight')
                feature = np.zeros([self.n,self.n])
                for i in range(self.n):
                    for j in range(self.n):
                        feature[i,j] = (eigen[i])*(eigen[j])
            elif mode =='r weighted eigen':
                eigen = nx.eigenvector_centrality_numpy(self.agg_graph, weight = 'weight')
                feature = np.zeros([self.n,self.n])
                for i in range(self.n):
                    for j in range(self.n):
                        feature[i,j] = 1/(eigen[i]*eigen[j])
            elif mode =='unweighted eigen':
                eigen = nx.eigenvector_centrality_numpy(self.agg_graph)
                feature = np.zeros([self.n,self.n])
                for i in range(self.n):
                    for j in range(self.n):
                        feature[i,j] = eigen[i]*eigen[j]
            elif mode =='r unweighted eigen':
                eigen = nx.eigenvector_centrality_numpy(self.agg_graph)
                feature = np.zeros([self.n,self.n])
                for i in range(self.n):
                    for j in range(self.n):
                        feature[i,j] = 1/(eigen[i]*eigen[j])
            elif mode =='random':
                feature = np.ones([self.n,self.n])
            else:
                print('unspported mode')
            block = feature*np.sum(self.backbones)/np.sum(feature*self.backbones)*block_prob
            while True:
                block[block>1] = 1
                new = np.sum(self.backbones)*block_prob-np.sum(self.backbones[block==1])
                old = np.sum((block*self.backbones)[block<1])
                block[block<1] = block[block<1]*new/old
                block[block>1] = 1
                if abs(np.sum(self.backbones*block)-np.sum(self.backbones)*block_prob)<0.01:
                    break
            self.block_prob[mode] = block
    
    def spread(self, betas=[0.01], repeat = 1):
        self.count = np.zeros([len(self.contacts), len(self.modes),len(betas),repeat,self.n])
        self.infect = np.zeros([len(self.modes),len(betas),repeat,self.n, self.n], dtype = bool)
        
        for i,(_,c) in tqdm(enumerate(self.contacts.iterrows())):
            for m,mode in enumerate(self.modes):
                for b,beta in enumerate(betas):
                    for r in range(repeat):
                        for node in range(self.n):
                            self.infect[m, b, r, node, node] = 1
                            t = c['time']
                            p1 = c['id1']
                            p2 = c['id2']
                            if self.block_prob[mode][p1,p2] < np.random.random():
                                if self.infect[m, b, r, node, p1] or self.infect[m, b, r, node, p2]:
                                    if beta > np.random.random():
                                        self.infect[m, b, r, node, p1] = 1
                                        self.infect[m, b, r, node, p2] = 1
            self.count[i] = np.mean(self.infect,axis = 4)
        return self.count

modes = ['degree product', 'r degree product','strength product','r strength product',
         'betweeness', 'r betweeness', 'random','link weight', 'r link weight',
         'weighted eigen','r weighted eigen','unweighted eigen','r unweighted eigen']

parser = argparse.ArgumentParser(description='Simulation')
parser.add_argument('--dataset', type=str, default='hs11',
                    help='datasets')
args = parser.parse_args()


s_model = model(datasets[args.dataset], modes)
infect = s_model.spread(repeat = 5)
for m,mode in enumerate(modes):
    np.save('results/'+args.dataset+'_'+mode+'_90%.npy',infect[:,m])