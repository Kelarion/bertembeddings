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
    LOAD_DIR = '/home/matteo/Documents/github/bertembeddings/data/extracted/miguels/'
    data_file = '/home/matteo/Documents/github/bertembeddings/data/extracted/phrase_boundary_tree_dist.pkl'
else:    
    CODE_DIR = '/home/malleman/bert_code/'
    LOAD_DIR = '/om2/user/drmiguel/CWR_manifolds/vectors/permuted_depth/bert-base-cased/'
    SAVE_DIR = '/om2/user/malleman/bert/miguel/'
    data_file = '/om2/user/drmiguel/CWR_manifolds/datasets/permuted_phrases/phrase_boundary_tree_dist.pkl'

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

#%%
allargs = sys.argv

arglist = allargs[1:]

# unixOpts = "vd:"
# gnuOpts = ["verbose"]

# opts, _ = getopt.getopt(arglist, unixOpts, gnuOpts)

# verbose, N, init = False, None, None # defaults
# for op, val in opts:
#     if op in ('-v','--verbose'):
#         verbose = True
#     if op in ('-d'):
#         tree_dist = int(val)

tree_dist = int(allargs[1])

folder = 'tree%d/'%tree_dist
# SAVE_DIR += folder

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
# these_tree_distances = [2,3,4,5] # which tree distances do we want to consider
max_sample = 400 # the maximum number of sample per condition 
pca_comp = 400

with open(data_file, 'rb') as dfile:
    dist = pkl.load(dfile)


# for i,d in enumerate(these_tree_distances):

SW = [] # swapped vectors
OG = [] # original vectors
which_toks = []
t_dist = [] # distance on parse tree
# s_dist = [] # distance in sequence (i.e. number of words between)
which_line = []

# num_cond = np.zeros(len(these_tree_distances))
num = 0
num_vecs = 0

these_lines = [i for i,d in enumerate(dist) if d[5]==tree_dist]
# these_lines =  list(open(SAVE_DIR+'miguel_lines_dist%d.txt'%tree_dist,'r').readlines())
# these_lines = [int(i[:-1]) for i in these_lines]

print('Beginning extraction ... ')
t0 = time()
for line_idx in np.random.permutation(these_lines): # range(13):
# for line_idx in np.random.permutation
    # print('Line %d'%line_idx)
    
    # if dist[line_idx][5] != tree_dist:
    #     print('Oops! Distances are wrong!')
    #     break
    
    try: 
        with open(LOAD_DIR+'/%d/original_vectors.pkl'%line_idx,'rb') as vfile:
            original_vectors = pkl.load(vfile)
    except FileNotFoundError:
        print('Skipping line, doesn"t exist'%line_idx)
        continue
    
    
    swap_idx = dist[line_idx][2]
    t = swap_idx[0]
    dt = np.abs(swap_idx[1]-swap_idx[0])
    
    
    try: 
        with open(LOAD_DIR+'/%d/swapped_vectors.pkl'%line_idx,'rb') as vfile:
            swapped_vectors = pkl.load(vfile)
    except FileNotFoundError:
        print('Skipping line %d, doesn"t exist'%line_idx)
        continue
        
    num += 1
    
    # shuffle = np.random.permutation(len(dist[line_idx][0]))
    # shuffle = np.arange(len(dist[line_idx][0]), dtype=int)
    # shuffle[np.array(swap_idx)] = np.flip(swap_idx)
    OG.append(original_vectors)
    SW.append(swapped_vectors)
    t_dist.append(dist[line_idx][5])
    which_line.append(np.repeat(line_idx,len(dist[line_idx][0])))
    which_toks.append(np.array(swap_idx)+num_vecs)
    num_vecs += len(dist[line_idx][0])
    
    if np.all(num>=max_sample):
        print('Done early!')
        break
    # else:
    #     print('Extracted %d/%d samples at %.3f seconds'%(num, max_sample, time()-t0))
    
    # print('Done with line %d in %.3f seconds'%(line_idx,time()-t0))

# do the CCA/cosine similarity
# OG = original_vectors[:,:,which_toks]
OG = np.concatenate(OG, axis=2) # shape (13,768,N)
SW = np.concatenate(SW, axis=2)
t_dist = np.array(t_dist)
which_line = np.concatenate(which_line)
swapped_toks = np.array(which_toks)

assert num_vecs==OG.shape[2]
# reformate which_toks to index the full array
# offset = np.append(0, np.cumsum(np.unique(which_line, return_counts=True)[1])[:-1])
# swapped_toks += offset[:,None]

print('Extracted %d vectors from %d lines, for tree dist %d'%(OG.shape[2], num, tree_dist))

# save the indexing arrays for future use
np.save(open(SAVE_DIR+folder+'line_number.npy','wb'), which_line)
np.save(open(SAVE_DIR+folder+'swap_indices.npy','wb'), swapped_toks)
np.save(open(SAVE_DIR+folder+'tree_distances.npy','wb'), t_dist)
# np.save(open(SAVE_DIR+'token_distances.npy','wb'), s_dist)

print('Computing cosines ...')
orig_centred = OG-OG.mean(axis=2, keepdims=True)
swap_centred = SW-SW.mean(axis=2, keepdims=True)
normalizer = (la.norm(orig_centred,2,1,keepdims=True)*la.norm(swap_centred,2,1,keepdims=True))
csim = np.sum((orig_centred*swap_centred)/normalizer,1)

np.save(open(SAVE_DIR+folder+'cosines_subsampled.npy','wb'), csim)


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

print('All done!!!!!!!!!!!!!!!!!')


