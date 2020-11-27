
import socket
if socket.gethostname() == 'kelarion':
    CODE_DIR = 'C:/Users/mmall/Documents/github/bertembeddings/'
    SAVE_DIR = 'C:/Users/mmall/Documents/github/bertembeddings/'
    LOAD_DIR = 'C:/Users/mmall/Documents/github/bertembeddings/data/extracted/'
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
    
import pickle as pkl
import bz2

import numpy as np
import scipy.linalg as la

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

#%% manually parse arguments
allargs = sys.argv

arglist = allargs[1:]


# these_tree_distances = [2,3,4,5] # which tree distances do we want to consider
# these_seq_distances = [1,2,3,4,5] # which sequence distances do we want to consider?
# tree_dist = int(np.mod(int(allargs[1]),4))+2
# seq_dist = int(np.floor(int(allargs[1])/4))+1
layer = int(np.mod(int(allargs[1]),13))
seq_dist = 1

tree_type = allargs[2].lower()
if tree_type=='dep':
    tree_dist = int(np.floor(int(allargs[1])/4))+1
    dep = True
    depth = False
    bracket_file = 'dependency_train_bracketed.txt'
    index_file = 'const_in_dep.npy' # to match indices in bracket_file
elif tree_type=='const':
    # tree_dist = int(np.floor(int(allargs[1])/4))+2
    dep = False
    depth = False
    bracket_file = 'train_bracketed.txt'
else:
    raise ValueError('`%s` is not a valid parse tree type'%tree_type)

init = int(allargs[1])//13
    
#%%
# def mypca(X, n):
#     """Assumes the features of X are along axis=1"""
#     U, S, _ = la.svd(X-X.mean(1, keepdims=True), full_matrices=False)
#     pcs = X@U[:n,:].T
#     var = np.sum((S[:n]**2)/np.sum(S**2))
    
#     return pcs, var

#%%
verbose = False

bsz = 50
nepoch = 10
lr = 1e-3

pos_tags = list(np.load(SAVE_DIR+'unique_pos_tags.npy'))
phrase_tags = list(np.load(SAVE_DIR+'unique_phrase_tags.npy'))

glm_orig_pos = nn.Linear(768, len(pos_tags), bias=True)
glm_swap_pos = nn.Linear(768, len(pos_tags), bias=True)
glm_orig_syn = nn.Linear(768, len(phrase_tags), bias=True)
glm_swap_syn = nn.Linear(768, len(phrase_tags), bias=True)

optimizer_orig_pos = optim.Adam(glm_orig_pos.parameters(),lr)
optimizer_swap_pos = optim.Adam(glm_swap_pos.parameters(),lr)
optimizer_orig_syn = optim.Adam(glm_orig_syn.parameters(),lr)
optimizer_swap_syn = optim.Adam(glm_swap_syn.parameters(),lr)

with open('/om3/group/chung/cca/datasets/permuted_phrases/swapped_data.pkl', 'rb') as dfile:
    dist = pkl.load(dfile)

if dep:
    idx = np.load(SAVE_DIR+'/data/'+index_file).astype(int)
else:
    idx = np.arange(len(dist))
 

train_loss_orig_pos = []
train_loss_swap_pos = []
train_loss_orig_syn = []
train_loss_swap_syn = []

num = 0
num_vecs = 0
print('Fitting GLMs ... ')
t0 = time()
for epoch in range(nepoch):
    nbatch = 0  # we'll only take gradient steps every bsz datapoints
    cumloss_orig_pos = 0
    cumloss_swap_pos = 0
    cumloss_orig_syn = 0
    cumloss_swap_syn = 0
    optimizer_orig_pos.zero_grad()
    optimizer_swap_pos.zero_grad()
    optimizer_orig_syn.zero_grad()
    optimizer_swap_syn.zero_grad()

    for line_idx in np.random.permutation(range(len(dist))): # range(13):
        
        line_idx_in_file = idx[line_idx]
        if line_idx_in_file < 0:
            continue
        line = linecache.getline(SAVE_DIR+'/data/'+bracket_file, line_idx_in_file+1)
        sentence = BracketedSentence(line, dep_tree=dep)
        
        # check that the sentence in swapped_data.pkl matches that in train.txt
        sameword = [sentence.words[i]==dist[line_idx][0][i] for i in range(sentence.ntok)]
        if ~np.all(sameword):
            if verbose:
                print('Mismatch between lines!!')
                print('Line %d is '%line_idx)
            break
        
        if sentence.ntok<10:
            if verbose:
                print('Skipping line %d, too short!'%line_idx)
            continue
        
        try:    
            with bz2.BZ2File(LOAD_DIR+'/%d/original_vectors.pkl'%line_idx,'rb') as vfile:
                original_vectors = pkl.load(vfile)
        except FileNotFoundError:
            if verbose:
                print('Skipping line %d, doesn"t exist'%line_idx)
            continue
        
        valid = np.array([w in pos_tags for w in sentence.pos_tags])
        valid_ = np.array([sentence.ancestor_tags(i,2) in phrase_tags for i in range(sentence.ntok)])
        
        # assumes the data were generated according to a certain algorithm
        # dt = np.repeat(np.arange(1,sentence.ntok-1), np.arange(sentence.ntok-1,1,-1)-1)
        # these_pairs = np.where(dt==seq_dist)[0]
        these_pairs = [i for i in range(len(dist[line_idx][1])) if np.diff(dist[line_idx][1][i][1])==seq_dist]
        
        running_list = [] # keep track of which tokens we've used
        # taken = np.zeros(len(num_cond))
        for pair in np.random.permutation(these_pairs):
            
            swap_idx = list(dist[line_idx][1][pair][1])
            if not (np.all(valid[swap_idx]) and np.all(valid_[swap_idx])):
                if verbose:
                    print('Neither word has a valid tag!!')
                continue
            try:
                if depth:
                    if not dep: # check if within phrase
                        if sentence.is_relative(swap_idx[0], swap_idx[1], order=1):
                            continue
                    d_tree = sentence.parse_depth(swap_idx[1], term=(not dep))\
                        -sentence.parse_depth(swap_idx[0], term=(not dep))
                    d_tree = int(np.abs(d_tree))
                else:
                    d_tree = sentence.tree_dist(swap_idx[0], swap_idx[1], term=(not dep))
            except IndexError as e:
                print(swap_idx)
                print(sentence.word2node)
                print(sentence.term2word)
                raise e
            
            if d_tree != tree_dist:
                continue
            
            t = swap_idx[0]
            s_diff = np.abs(swap_idx[1]-swap_idx[0])
            if s_diff != seq_dist:
                print('Oops, not right')
                continue
            
            # if np.sum(np.isin(swap_idx, running_list)):
            #     continue
            
            try: 
                with bz2.BZ2File(LOAD_DIR+'/%d/%d_swapped_vectors.pkl'%(line_idx, pair),'rb') as vfile:
                    swapped_vectors = pkl.load(vfile)
            except FileNotFoundError:
                if verbose:
                    print('Skipping line %d, pair %d, doesn"t exist'%(line_idx, pair))
                continue
            except OSError:
                if verbose:
                    print('Problem with line %d pair %d'%(line_idx, pair))
                continue
            except EOFError:
                if verbose:
                    print('\_(`v`)_/')
                continue
            
            ## compute the loss
            labels_pos = np.array([pos_tags.index(w) if w in pos_tags else np.nan for w in sentence.pos_tags])[swap_idx]
            y_pos = torch.tensor(labels_pos).long()
            
            anc_tags = [sentence.ancestor_tags(i,2) for i in range(sentence.ntok)]
            labels_syn = np.array([phrase_tags.index(w) if w in phrase_tags else np.nan for w in anc_tags])[swap_idx]
            y_syn = torch.tensor(labels_syn).long()
            
            X_orig = torch.tensor(original_vectors[layer,:,swap_idx])
            loss_orig_pos = nn.CrossEntropyLoss()(glm_orig_pos(X_orig), y_pos)
            loss_orig_syn = nn.CrossEntropyLoss()(glm_orig_syn(X_orig), y_syn)
            
            X_swap = torch.tensor(swapped_vectors[layer,:,swap_idx])
            loss_swap_pos = nn.CrossEntropyLoss()(glm_swap_pos(X_swap), y_pos)
            loss_swap_syn = nn.CrossEntropyLoss()(glm_swap_syn(X_swap), y_syn)
            
            ## do the batc advancement
            if nbatch<bsz: # still in batch
                cumloss_orig_pos += loss_orig_pos
                cumloss_swap_pos += loss_swap_pos
                cumloss_orig_syn += loss_orig_syn
                cumloss_swap_syn += loss_swap_syn
                nbatch+=1
            else: # end of batch
                train_loss_orig_pos.append(cumloss_orig_pos.item())
                train_loss_swap_pos.append(cumloss_swap_pos.item())
                train_loss_orig_syn.append(cumloss_orig_syn.item())
                train_loss_swap_syn.append(cumloss_swap_syn.item())
                cumloss_orig_pos.backward()
                cumloss_swap_pos.backward()
                cumloss_orig_syn.backward()
                cumloss_swap_syn.backward()
                optimizer_orig_pos.step()
                optimizer_swap_pos.step()
                optimizer_orig_syn.step()
                optimizer_swap_syn.step()
                
                nbatch = 0
                cumloss_orig_pos = 0
                cumloss_swap_pos = 0
                cumloss_orig_syn = 0
                cumloss_swap_syn = 0
                optimizer_orig_pos.zero_grad()
                optimizer_swap_pos.zero_grad()
                optimizer_orig_syn.zero_grad()
                optimizer_swap_syn.zero_grad()

#%%
fold = 'linear_probes/'

if not os.path.isdir(SAVE_DIR+fold):
    os.makedirs(SAVE_DIR+fold)
    
np.save(SAVE_DIR+fold+'orig_pos_layer%d_tree%d_train_loss.npy'%(layer,tree_dist), train_loss_orig_pos)
np.save(SAVE_DIR+fold+'swap_pos_layer%d_tree%d_train_loss.npy'%(layer,tree_dist), train_loss_swap_pos)
np.save(SAVE_DIR+fold+'orig_syn_layer%d_tree%d_train_loss.npy'%(layer,tree_dist), train_loss_orig_syn)
np.save(SAVE_DIR+fold+'swap_syn_layer%d_tree%d_train_loss.npy'%(layer,tree_dist), train_loss_swap_syn)
with open(SAVE_DIR+fold+'orig_pos_layer%d_tree%d_params.pt'%(layer,tree_dist), 'wb') as f:
    torch.save(glm_orig_pos.state_dict(), f)
with open(SAVE_DIR+fold+'swap_pos_layer%d_tree%d_params.pt'%(layer,tree_dist), 'wb') as f:
    torch.save(glm_swap_pos.state_dict(), f)
with open(SAVE_DIR+fold+'orig_syn_layer%d_tree%d_params.pt'%(layer,tree_dist), 'wb') as f:
    torch.save(glm_orig_syn.state_dict(), f)
with open(SAVE_DIR+fold+'swap_syn_layer%d_tree%d_params.pt'%(layer,tree_dist), 'wb') as f:
    torch.save(glm_orig_syn.state_dict(), f)


