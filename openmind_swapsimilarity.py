#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 17:57:38 2020

@author: matteo
"""

import socket
if socket.gethostname() == 'kelarion':
    CODE_DIR = '/home/matteo/Documents/github/bertembeddings/'
    SAVE_DIR = '/home/matteo/Documents/github/bertembeddings/'
    LOAD_DIR = '/home/matteo/Documents/github/bertembeddings/data/extracted/'
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
seq_dist = int(np.mod(int(allargs[1]),4))+1
tree_dist = int(np.floor(int(allargs[1])/4))+1


tree_type = allargs[2].lower()
if tree_type=='dep':
    seq_dist = int(np.mod(int(allargs[1]),4))+1
    tree_dist = int(np.floor(int(allargs[1])/4))+1
    dep = True
    depth = False
    bracket_file = 'dependency_train_bracketed.txt'
    index_file = 'const_in_dep.npy' # to match indices in bracket_file
elif tree_type=='depdepth':
    dep = True
    depth = True
    seq_dist = int(np.mod(int(allargs[1]),4))+1
    tree_dist = int(np.floor(int(allargs[1])/4))
    bracket_file = 'dependency_train_bracketed.txt'
    index_file = 'const_in_dep.npy' # to match indices in bracket_file
elif tree_type=='const':
    seq_dist = int(np.mod(int(allargs[1]),4))+1
    tree_dist = int(np.floor(int(allargs[1])/4))+2
    dep = False
    depth = False
    bracket_file = 'train_bracketed.txt'
elif tree_type=='constdepth':
    seq_dist = int(np.mod(int(allargs[1]),4))+1
    tree_dist = int(np.floor(int(allargs[1])/4))
    dep = False
    depth = True
    bracket_file = 'train_bracketed.txt'
else:
    raise ValueError('`%s` is not a valid parse tree type'%tree_type)

folder = '%s/tree%d-seq%d/'%(tree_type, tree_dist, seq_dist)

if not os.path.isdir(SAVE_DIR+folder):
    os.makedirs(SAVE_DIR+folder)
    
#%%
# def mypca(X, n):
#     """Assumes the features of X are along axis=1"""
#     U, S, _ = la.svd(X-X.mean(1, keepdims=True), full_matrices=False)
#     pcs = X@U[:n,:].T
#     var = np.sum((S[:n]**2)/np.sum(S**2))
    
#     return pcs, var

#%%
max_sample = 400 # the maximum number of sample per condition 
pca_comp = 400
do_cca = False
verbose = False

with open('/om3/group/chung/cca/datasets/permuted_phrases/swapped_data.pkl', 'rb') as dfile:
    dist = pkl.load(dfile)

if dep:
    idx = np.load(SAVE_DIR+'/data/'+index_file).astype(int)
else:
    idx = np.arange(len(dist))
 
SW = [] # swapped vectors
OG = [] # original vectors
which_toks = []
# t_dist = [] # distance on parse tree
# s_dist = [] # distance in sequence (i.e. number of words between)
which_line = [] # outside index of swapped_data
which_swap = [] # inside index of swapped_data

num = 0
num_vecs = 0
print('Beginning extraction ... ')
t0 = time()
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
    
    # assumes the data were generated according to a certain algorithm
    # dt = np.repeat(np.arange(1,sentence.ntok-1), np.arange(sentence.ntok-1,1,-1)-1)
    # these_pairs = np.where(dt==seq_dist)[0]
    these_pairs = [i for i in range(len(dist[line_idx][1])) if np.diff(dist[line_idx][1][i][1])==seq_dist]
    
    running_list = [] # keep track of which tokens we've used
    # taken = np.zeros(len(num_cond))
    for pair in np.random.permutation(these_pairs):
        
        swap_idx = dist[line_idx][1][pair][1]
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
        

        # running_list.append(swap_idx)
        num += 1
    
        OG.append(original_vectors)
        SW.append(swapped_vectors)
        # t_dist.append(d_tree)
        which_line.append(np.repeat(line_idx,len(dist[line_idx][0])))
        which_swap.append(np.repeat(num,len(dist[line_idx][0])))
        which_toks.append(np.array(swap_idx)+num_vecs)
        num_vecs += len(dist[line_idx][0])
        
        if num>=400:
            print('Done early!')
            break
        # else:
    if num >= 400:
        break
    if verbose:
        print('Extracted %d/%d samples at %.3f seconds'%(num, max_sample, time()-t0))


# do the CCA/cosine similarity
OG = np.concatenate(OG, axis=2) # shape (13,768,N)
SW = np.concatenate(SW, axis=2)
# t_dist = np.array(t_dist)
which_line = np.concatenate(which_line)
which_swap = np.concatenate(which_swap)
swapped_toks = np.array(which_toks)

assert num_vecs==OG.shape[2]
# reformate which_toks to index the full array
# offset = np.append(0, np.cumsum(np.unique(which_line, return_counts=True)[1])[:-1])
# swapped_toks += offset[:,None]

print('Extracted %d vectors from %d lines, for tree dist %d'%(OG.shape[2], num, tree_dist))

# save the indexing arrays for future use
np.save(open(SAVE_DIR+folder+'line_number.npy','wb'), which_line)
np.save(open(SAVE_DIR+folder+'swap_number.npy','wb'), which_swap)
np.save(open(SAVE_DIR+folder+'swap_indices.npy','wb'), swapped_toks)
# np.save(open(SAVE_DIR+folder+'tree_distances.npy','wb'), t_dist)
# np.save(open(SAVE_DIR+'token_distances.npy','wb'), s_dist)

print('Computing cosines ...')
orig_centred = OG-OG.mean(axis=2, keepdims=True)
swap_centred = SW-SW.mean(axis=2, keepdims=True)
normalizer = (la.norm(orig_centred,2,1,keepdims=True)*la.norm(swap_centred,2,1,keepdims=True))
csim = np.sum((orig_centred*swap_centred)/normalizer,1)

np.save(open(SAVE_DIR+folder+'cosines_subsampled.npy','wb'), csim)


print('Computing matrix norms ...')
frob = [la.norm(orig_centred[:,:,which_swap==l]-swap_centred[:,:,which_swap==l],'fro',axis=(1,2)) \
        for l in np.unique(which_swap)]
nuc = [la.norm(orig_centred[:,:,which_swap==l]-swap_centred[:,:,which_swap==l], 'nuc',axis=(1,2)) \
       for l in np.unique(which_swap)]
inf = [la.norm(orig_centred[:,:,which_swap==l]-swap_centred[:,:,which_swap==l], np.inf,axis=(1,2)) \
       for l in np.unique(which_swap)]
    
# geometric quantities -- participation ratio and cosines
diff = OG-SW
diff -= diff.mean(1, keepdims=True)
lindim = []
parallel = []
for l in range(13):
    dots = [diff[l,:,which_swap==s].T@diff[l,:,which_swap==s] \
            for s in np.unique(which_swap)]
    eigs = [la.svd(diff[l,:,which_swap==s], full_matrices=False)[1]**2 \
            for s in np.unique(which_swap)]
    lindim.append([np.sum(e)**2 / np.sum(e**2) for e in eigs])
    parallel.append([d[np.triu(d)!=0].mean() for d in dots])

np.save(open(SAVE_DIR+folder+'swap_frob_distance.npy','wb'), frob)
np.save(open(SAVE_DIR+folder+'swap_nuc_distance.npy','wb'), nuc)
np.save(open(SAVE_DIR+folder+'swap_inf_distance.npy','wb'), inf)
np.save(open(SAVE_DIR+folder+'difference_dimension.npy','wb'), lindim)
np.save(open(SAVE_DIR+folder+'difference_parallelism.npy','wb'), parallel)

print('Computing distances ...')
l2dist = la.norm(OG-SW, 2, axis=1)
np.save(open(SAVE_DIR+folder+'swap_distances.npy','wb'), l2dist)

og_dist = la.norm(np.diff(OG[:,:,swapped_toks.T], axis=2), 2, axis=1).squeeze()
sw_dist = la.norm(np.diff(SW[:,:,swapped_toks.T], axis=2), 2, axis=1).squeeze()
np.save(open(SAVE_DIR+folder+'original_distances.npy','wb'), og_dist)
np.save(open(SAVE_DIR+folder+'swapped_distances.npy','wb'), sw_dist)


# tree = np.tile(t_dist, 2)
# sent = np.tile(s_dist, 2)
# nsamp = OG.shape[-1]*2
# print(OG.shape)
if do_cca and (num_vecs>pca_comp):
    print('Computing CCA ...')
    # CCA = np.zeros(13)
    # CCA_err = np.zeros(13)
    CCA = []
    CCA_swapped = []
    CCA_weights = []
    explained = np.zeros(13)
    rank = np.zeros(13)
    for l in range(13):
        print('... layer %d... '%l)
        together = np.append(OG[l,:,:], SW[l,:,:], axis=-1).T
        
        # a hack to make sure the matrices aren't rank decifient
        n_pca = np.min([np.linalg.matrix_rank(OG[l,:,:]), pca_comp])
        pca = PCA(n_components=n_pca)
        print('Using %d components'%n_pca)
        pca.fit(together)    
        M1 = pca.transform(OG[l,:,:].T).T
        M2 = pca.transform(SW[l,:,:].T).T
        
        _, weights, coefs = compute_pwcca(M1, M2)
        CCA.append(coefs)
        CCA_weights.append(weights)
        
        # only swapped tokens
        M1 = pca.transform(OG[l,:,swapped_toks.flatten()]).T
        M2 = pca.transform(SW[l,:,swapped_toks.flatten()]).T
        
        _, weights, coefs = compute_pwcca(M1, M2)
        CCA_swapped.append(coefs)
        
        # CCA[l,i,j] = np.mean(get_cca_similarity(M1[:,cond], M2[:,cond])['cca_coef1'])
        print('Explained %.3f variance'%np.sum(pca.explained_variance_ratio_))
        explained[l] = np.sum(pca.explained_variance_ratio_)
        rank[l] = n_pca
    
    pkl.dump(CCA_swapped, open(SAVE_DIR+folder+'cca_swp.pkl','wb'))
    pkl.dump(CCA, open(SAVE_DIR+folder+'cca_coefs.pkl','wb'))
    pkl.dump(CCA_weights, open(SAVE_DIR+folder+'cca_weights.pkl','wb'))
    # np.save(open(SAVE_DIR+'cca_subsampled.npy','wb'), CCA)
    # np.save(open(SAVE_DIR+'cca_errors.npy','wb'), CCA_err)
    np.save(open(SAVE_DIR+folder+'pca_explained.npy','wb'), explained) 
    np.save(open(SAVE_DIR+folder+'pca_rank.npy','wb'), rank)
else:
    print('Too few vectors, skipping CCA!')

print('All done!!!!!!!!!!!!!!!!!')



