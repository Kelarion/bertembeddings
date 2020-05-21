#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 09:10:11 2020

@author: matteo
"""

CODE_DIR = '/home/matteo/Documents/github/bertembeddings/'
SAVE_DIR = '/home/matteo/Documents/github/bertembeddings/data/bert/'
LOAD_DIR = '/home/matteo/Documents/github/bertembeddings/data/extracted/'

import sys
# sys.path.append(CODE_DIR+'repler/src/')

sys.path.append(CODE_DIR)
from brackets2trees import BracketedSentence
from pwcca import compute_pwcca
from cca_core import get_cca_similarity
from hyperbolic_utils import CartesianHyperboloid, EuclideanEncoder, GeodesicCoordinates

import pickle as pkl
import bz2

import numpy as np
import scipy.linalg as la
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import networkx as nx

import os
import csv
import random
from time import time
import getopt
import linecache

from sklearn.decomposition import PCA

from itertools import permutations

#%%
N = 2
layer = 8
# euc = True
euc = False
# dep = True
dep = False

# criterion = nn.MSELoss(reduction='mean')
criterion = nn.L1Loss(reduction='mean')
# criterion = nn.PoissonNLLLoss(reduction='mean', log_input=False)

encoder = nn.Linear(768, N, bias=False)

index_file = 'const_in_dep.npy' 

if dep:
    idx = np.load(SAVE_DIR+index_file).astype(int)
    bracket_file = 'dependency_train_bracketed.txt'
    fold = 'dep/'
else:
    idx = np.arange(3001)
    bracket_file = 'train_bracketed.txt'
    fold = 'const'
    # idx = np.arange(len(dist))

if euc:
    probe = EuclideanEncoder(encoder)
else:
    probe = GeodesicCoordinates(encoder)

# init = probe.normalize(2*1e-3*(torch.rand(768, N+1)-0.5))
# probe.enc.weight.data = probe.invchart(init)
# probe.init_weights()

# bracket_file = 'dependency_train_bracketed.txt'

expinf = 'layer%d_rank%d_%s_linear'%(layer, N, criterion.__class__.__name__)

if euc:
    param_file = '/EuclideanEncoder/'+expinf+'_params.pt'
else:
    param_file = '/GeodesicCoordinates/'+expinf+'_params.pt'
probe.load_state_dict(torch.load(SAVE_DIR+fold+param_file))


    #%%
means = []
std = []
for line_idx in [2]:
    ############## load the sentence
    line_idx_in_file = idx[line_idx]
    if line_idx_in_file < 0:
        continue
    line = linecache.getline(SAVE_DIR+bracket_file, line_idx_in_file+1)
    sentence = BracketedSentence(line, dep_tree=dep)
    
    # check that the sentence in swapped_data.pkl matches that in train.txt
    # sameword = [sentence.words[i]==dist[line_idx][0][i] for i in range(sentence.ntok)]
    # if ~np.all(sameword):
    #     print('Mismatch between lines!!')
    #     print('Line %d is '%line_idx)
    #     break
    
    if sentence.ntok<10:
        print('Skipping line %d, too short!'%line_idx)
        continue
    
    try: 
        with bz2.BZ2File(LOAD_DIR+'/%d/original_vectors.pkl'%line_idx,'rb') as vfile:
            original_vectors = pkl.load(vfile)
    except FileNotFoundError:
        print('Skipping line %d, doesn"t exist'%line_idx)
        continue
    
    toks = np.arange(sentence.ntok)
    ntok = sentence.ntok
    w1, w2 = np.nonzero(np.triu(np.ones((ntok,ntok)),k=1))
    
    # probe distance
    W1 = probe(torch.tensor(original_vectors[layer,:,w1])) # map onto euclidean or hyperbolic space
    W2 = probe(torch.tensor(original_vectors[layer,:,w2]))
    
    dB = probe.dist(W1, W2)
    
    # tree distance
    dT = torch.tensor([sentence.tree_dist(w1[i],w2[i],term=(not dep)) for i in range(len(w1))]).float()
    means.append(dT.mean().numpy())
    std.append(dT.var().numpy())
    

#%%

wa = probe(torch.tensor(original_vectors[layer,:,:].T)).detach()
# poinc = wa
poinc = wa[:,1:]/(1+wa[:,:1])

pos = {i:poinc[sentence.node2word[i],:].numpy() for i in range(sentence.ntok)}
    
grph = nx.Graph()
grph.add_edges_from(sentence.edges)
nx.draw_networkx(grph, pos, with_labels=False, node_size=20)
# plt.scatter(poinc[:,0].detach(),poinc[:,1].detach())

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

for i in range(sentence.ntok):
    plt.text(poinc[i,0], poinc[i,1], sentence.words[i], bbox=props)
    

plt.xlim([-1,1])
plt.ylim([-1,1])

plt.axis('equal')
