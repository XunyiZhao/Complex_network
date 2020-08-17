import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt

# Read datasets
hs11 = pd.read_csv('datasets/thiers_2011.csv', delimiter='\t',names = ['time','p1','p2','c1','c2'])[['time','p1','p2']]
hs12 = pd.read_csv('datasets/thiers_2012.csv', delimiter=',',names = ['time','p1','p2','c1','c2'])[['time','p1','p2']]
datasets = {'hs11':hs11,'hs12':hs12}

# Define model
class model():
    def __init__(self, contacts, mode):
        self.contacts = contacts
        self.people = np.unique(self.contacts[['p1','p2']])
        self.n = len(self.people)
        p2id = {}
        for i, p in enumerate(self.people):
            p2id[p] = i
        self.contacts['id1'] = self.contacts['p1'].apply (lambda row: p2id[row])
        self.contacts['id2'] = self.contacts['p2'].apply (lambda row: p2id[row])

        # Aggregate the contacts
        self.backbones = np.zeros([self.n,self.n])
        for i, c in self.contacts.iterrows():
            t,p1,p2 = c[['time','id1','id2']]
            self.backbones[p1,p2] += 1
            self.backbones[p2,p1] += 1

        self.agg_graph = nx.from_numpy_array(self.backbones)
        
        if mode =='degree product':
            adj = (self.backbones>0)
            degree = np.sum(adj,axis=0)
            self.metric = np.outer(degree, degree)
        elif mode =='r degree product':
            adj = (self.backbones>0)
            degree = np.sum(adj,axis=0)
            self.metric = 1/np.outer(degree, degree)
        elif mode =='strength product':
            strength = np.sum(self.backbones,axis=0)
            self.metric = np.outer(strength, strength)
        elif mode =='r strength product':
            strength = np.sum(self.backbones,axis=0)
            self.metric = 1/np.outer(strength, strength)
        elif mode =='betweeness':
            self.metric = np.zeros([self.n,self.n])
            bet = nx.algorithms.edge_betweenness_centrality(self.agg_graph)
            for k,v in bet.items():
                self.metric[k] = v
            self.metric += self.metric.T
        elif mode =='r betweeness':
            self.metric = np.zeros([self.n,self.n])
            bet = nx.algorithms.edge_betweenness_centrality(self.agg_graph)
            for k,v in bet.items():
                self.metric[k] = 1/v
            self.metric += self.metric.T
        elif mode =='link weight':
            self.metric = np.copy(self.backbones)
        elif mode =='weighted eigen':
            eigen = nx.eigenvector_centrality_numpy(m.agg_graph, weight = 'weight')
            self.metric = np.zeros([self.n,self.n])
            for i in range(self.n):
                for j in range(self.n):
                    self.metric[i,j] = (eigen[i])*(eigen[j])
        elif mode =='unweighted eigen':
            eigen = nx.eigenvector_centrality_numpy(m.agg_graph)
            self.metric = np.zeros([self.n,self.n])
            for i in range(self.n):
                for j in range(self.n):
                    self.metric[i,j] = eigen[i]*eigen[j]
        elif mode =='random':
            self.metric = np.ones([self.n,self.n])
        else:
            print('unspported mode')
            
    
    def spread(self, betas=[0.01], block_prob = 0, repeat = 1):
        block = self.metric*np.sum(self.backbones)/np.sum(self.metric*self.backbones)*block_prob
        # To make sure probability<=1
        while True:
            block[block>1] = 1
            new = np.sum(self.backbones)*block_prob-np.sum(self.backbones[block==1])
            old = np.sum((block*self.backbones)[block<1])
            block[block<1] = block[block<1]*new/old
            block[block>1] = 1
            if abs(np.sum(self.backbones*block)-np.sum(self.backbones)*block_prob)<0.01:
                break
        count = np.zeros([len(betas),repeat,self.n, len(self.contacts)])
        
        for b,beta in enumerate(betas):
            for r in range(repeat):
                for node in range(self.n):
                    infect = np.zeros(self.n)
                    infect[node] = 1
                    for i,c in self.contacts.iterrows():
                        t = c['time']
                        p1 = c['id1']
                        p2 = c['id2']
                        if block[p1,p2] < np.random.random():
                            if infect[p1] or infect[p2]:
                                if beta > np.random.random():
                                    infect[p1] = 1
                                    infect[p2] = 1
                        count[b,r,node,i] += np.mean(infect)
        return count

# Run simulation
strategies = ['strength product','r strength product']
betas = [0.01]
block_prob = 0.1
repeat = 5

for name,data in datasets.items():
    for mode in tqdm(strategies):
        x = model(data, mode = mode)
        c = x.spread(betas = betas, block_prob = block_prob, repeat = repeat)
        np.save(mode+'_'+name,c)
        plt.plot(np.mean(c[0,:,:,:], axis=(0,1)),label=mode)
    plt.xlabel('step')
    plt.ylabel('infect rate')
    plt.title('90% contacts')
    plt.legend()
    plt.savefig(name+'90%contacts.png',dpi=400)