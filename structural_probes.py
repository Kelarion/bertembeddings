import socket
if socket.gethostname() == 'kelarion':
    CODE_DIR = '/home/matteo/Documents/github/bertembeddings/'
    SAVE_DIR = '/home/matteo/Documents/github/bertembeddings/data/'
    LOAD_DIR = '/home/matteo/Documents/github/bertembeddings/data/extracted'
else:    
    CODE_DIR = '/home/malleman/bert_code/'
    LOAD_DIR = '/om3/group/chung/cca/vectors/swapped/bert-base-cased/'
    SAVE_DIR = '/om2/user/malleman/bert/'

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
import scipy.stats as sts
import torch
import torch.nn as nn
import torch.optim as optim

import os
import csv
import random
from time import time
import getopt
import linecache

from sklearn.decomposition import PCA

from itertools import permutations

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

 #%%
allargs = sys.argv

arglist = allargs[1:]

N = int(allargs[1])

tree_type = allargs[2].lower()
if tree_type=='dep':
    dep = True
    bracket_file = 'dependency_train_bracketed.txt'
    index_file = 'const_in_dep.npy' # to match indices in bracket_file
elif tree_type=='const':
    dep = False
    bracket_file = 'train_bracketed.txt'
else:
    raise ValueError('%s is not a valid parse tree type'%tree_type)

svfolder = '%s/'%(tree_type)

if not os.path.isdir(SAVE_DIR+svfolder):
    os.makedirs(SAVE_DIR+svfolder)

#%%
# N = 2
bsz = 20
nepoch = 40
encoder = nn.Linear(768, N, bias=False)
hype = False

with open('/om3/group/chung/cca/datasets/permuted_phrases/swapped_data.pkl', 'rb') as dfile:
    dist = pkl.load(dfile)

if dep:
    idx = np.load(SAVE_DIR+'/data/'+index_file).astype(int)
else:
    idx = np.arange(len(dist))

# probe = EuclideanEncoder(encoder)
# probe = CartesianHyperboloid(encoder)
if hype:
    probe = GeodesicCoordinates(encoder)
    init = probe.normalize(2*1e-3*(torch.rand(768, N+1)-0.5))
    probe.enc.weight.data = probe.invchart(init)
    lr = 1e-4  # same parameters as Hewitt and Manning
    optimizer = optim.SGD(probe.parameters(),lr)
    burnin = 10
else:
    probe = EuclideanEncoder(encoder)
    lr = 1e-3  # same parameters as Hewitt and Manning
    optimizer = optim.Adam(probe.parameters(),lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=0)
    
# probe.init_weights()

# train_set = list(range(int(2*len(dist)/3)))
# test_set = list(range(int(2*len(dist)/3), len(dist)))

#%% Train
layer = 8

within_range = [i for i in range(len(dist)) if (len(dist[i][0])>=10)&(len(dist[i][0])<=110)]
train_set = within_range[:int(2*len(within_range)/3)]
test_set = within_range[int(2*len(within_range)/3):]

t0 = time()

distances = [[] for _ in range(len(dist))] # store the dTs, as they take some time to compute
train_loss = []
prev_weights = probe.state_dict()
for epoch in range(nepoch):
    if hype:
        if epoch < burnin:
            optimizer.param_groups[0]['lr'] = lr/10
        else:
            optimizer.param_groups[0]['lr'] = lr
    running_loss = 0
    nbatch = 0  # we'll only take gradient steps every bsz datapoints
    cumloss = 0
    optimizer.zero_grad()
    for line_idx in np.random.permutation(train_set): # range(13):
        ############## load the sentence
        line_idx_in_file = idx[line_idx]
        if line_idx_in_file < 0:
            continue
        line = linecache.getline(SAVE_DIR+'/data/'+bracket_file, line_idx_in_file+1)
        sentence = BracketedSentence(line, dep_tree=dep)
        
        # check that the sentence in swapped_data.pkl matches that in train.txt
        sameword = [sentence.words[i]==dist[line_idx][0][i] for i in range(sentence.ntok)]
        if ~np.all(sameword):
            print('Mismatch between lines!!')
            print('Line %d is '%line_idx)
            break
        
        if sentence.ntok<10:
            # print('Skipping line %d, too short!'%line_idx)
            continue
        if sentence.ntok>110:
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
        if epoch==0:
            dT_ = [sentence.tree_dist(w1[i],w2[i],term=(not dep)) for i in range(len(w1))]
            distances[line_idx] = dT_
            
        dT = torch.tensor(distances[line_idx]).float()
        
        loss = nn.L1Loss(reduction='mean')(dB, dT.expand_as(dB))
        running_loss += loss.item()
        
        naan = not (torch.all(probe.enc.weight.data==probe.enc.weight.data) or torch.all(W1==W1))
        if naan: # Contains NaN
            # probe.load_state_dict(prev_weights)
            print('Oops, NaN! at %d epochs'%epoch)
            naan = True
            break
        else:
            prev_weights = probe.state_dict()
        
        if nbatch<bsz: # still in batch
            cumloss += loss
            nbatch+=1
        else: # end of batch
            train_loss.append(cumloss.item())
            cumloss.backward()
            optimizer.step()
            cumloss = 0
            nbatch = 0
            optimizer.zero_grad()

    if naan:
        break
            
    # train_loss[epoch] = running_loss/(line_idx+1)
    print('Epoch %d: loss=%.3f'%(epoch, running_loss/(line_idx+1)))
    scheduler.step(running_loss/(line_idx+1))


folder = probe.__class__.__name__ + '/'
if not os.path.isdir(SAVE_DIR+svfolder+folder):
    os.makedirs(SAVE_DIR+svfolder+folder)

np.save(open(SAVE_DIR+svfolder+folder+'layer%d_rank%d_linear_train_idx.npy'%(layer, N),'wb'),train_set)
np.save(open(SAVE_DIR+svfolder+folder+'layer%d_rank%d_linear_test_idx.npy'%(layer, N),'wb'),test_set)

with open(SAVE_DIR+svfolder+folder+'layer%d_rank%d_linear_params.pt'%(layer, N),'wb') as f:
    torch.save(prev_weights,f)
np.save(open(SAVE_DIR+svfolder+folder+'layer%d_rank%d_linear_loss.npy'%(layer, N),'wb'),np.array(train_loss))


#%% test
# spr_per_length = np.zeros(100) # lengths {10, ..., 110}
dB_test = [np.zeros(0) for _ in range(100)]
dT_test = [np.zeros(0) for _ in range(100)]

for line_idx in np.random.permutation(test_set): # range(13):
    ############## load the sentence
    line_idx_in_file = idx[line_idx]
    if line_idx_in_file < 0:
        continue
    line = linecache.getline(SAVE_DIR+'/data/'+bracket_file, line_idx_in_file+1)
    sentence = BracketedSentence(line, dep_tree=dep)
    
    # check that the sentence in swapped_data.pkl matches that in train.txt
    sameword = [sentence.words[i]==dist[line_idx][0][i] for i in range(sentence.ntok)]
    if ~np.all(sameword):
        print('Mismatch between lines!!')
        print('Line %d is '%line_idx)
        break
    
    if sentence.ntok<10:
        # print('Skipping line %d, too short!'%line_idx)
        continue
    if sentence.ntok>110:
        continue
    len_idx = sentence.ntok+10
    
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
    dB_test[len_idx] = np.append(dB_test[len_idx], dB.detach().numpy())
    
    # tree distance
    dT = torch.tensor([sentence.tree_dist(w1[i],w2[i],term=(not dep)) for i in range(len(w1))]).float()
    
    dT_test[len_idx] = np.append(dT_test[len_idx], dT.detach().numpy())
    
    
spr_per_length = [sts.spearmanr(dT_test[i], dB_test[i])[0] for i in range(100)]
np.save(open(SAVE_DIR+svfolder+folder+'layer%d_rank%d_linear_dspr.npy'%(layer, N),'wb'),spr_per_length)


