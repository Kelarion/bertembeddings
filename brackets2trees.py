import sys

import pickle as pkl

import numpy as np
import scipy.linalg as la
import torch

import os
import re
import random
from time import time

#%% 
class BracketedSentence(object):
    def __init__(self, bs, dep_tree=False):
        """
        Take a bracketed sentence, bs, and find the edges of the corresponding
        parse tree and the token strings 
        
        If the brackets represent a dependency parse, set `dep_tree` to be True 
        """
        
        super(BracketedSentence,self).__init__()
        
        op = np.array([i for i,j in enumerate(bs) if j=='('])
        cl = np.array([i for i,j in enumerate(bs) if j==')'])
        braks = np.sort(np.append(op,cl))
        
        N_node = len(op)
        
        # First: find children
        iscl = np.isin(braks,cl) # label brackets
        
        term_op = braks[np.diff(iscl.astype(int),append=0)==1]
        
        assert np.all(np.isin(term_op,op)), "Unequal open and closed brackets!"
        
        leaves = np.where(np.isin(op,term_op))[0]
        # parse depth of each token is the total number of unclosed brackets
        depth = np.cumsum(~iscl*2-1)[np.isin(braks,op)]-1 
        
        # algorithm for finding all edges
        # takes advantage of the fact that nodes only have one parent in a tree
        nodes = np.arange(N_node)
        drawn = list(leaves)
        edges = []
        labels = ['' for _ in range(N_node)]
        i = 0
        while ~np.all(np.isin(nodes,drawn[:i])):
            # find edge
            if drawn[i]>0: # start token has no parent
                parent = np.where((op<op[drawn[i]]) & (depth==(depth[drawn[i]]-1)))[0][-1]
                edges.append((parent, drawn[i]))
            if parent not in drawn:
                drawn.append(parent)
                
            # get token string
            tok_end = braks[np.argmax(braks>op[drawn[i]])]
            tok = bs[op[drawn[i]]:tok_end]
            labels[drawn[i]] = tok.replace('(','')#.replace(' ','')
            
            i += 1
            
        self.edges = edges
        self.nodes = labels
        self.terminals = leaves # node indices of terminal tokens
        
        self.depth = depth

        # For computing the tree distances
        # get the brackets which are troughs in the parse depth 
        is_trough = np.flip(np.diff(np.flip(~iscl*2-1),prepend=1)) == -2
        trough_depths = np.cumsum(~iscl*2-1)[is_trough]-1
        is_trough[-1] = False
        # the depth of a particular token's subtree is the depth of the 
        subtree_depths = trough_depths[np.cumsum(is_trough)][np.isin(braks,op)]
        self.subtree_depth = subtree_depths
        
        # 
        if dep_tree:
            line = [n.split(' ')[1] for n in labels]
        else:
            words = np.array(labels)[leaves]
            line = [w.split(' ' )[1] for w in words]
        self.words = line
        self.ntok = len(line)
        
        self.bracketed = bs
        
        # when dealing with dep trees, the index in the actual sentence
        # is no the same as the index in the bracketed sentence 
        if dep_tree:
            node_names = [int(n.split(' ')[2]) for n in labels]
            order = list(np.argsort(node_names))
            self.words = [self.words[i] for i in order]
            word_names = [node_names.index(i) for i in range(self.ntok)]
        else:
            node_names = list(range(len(labels)))
            word_names = list(range(len(labels)))
            
        self.node2word = node_names # indices of each node in the sentence
        self.word2node = word_names # indices of each word in the node list
        
    def __repr__(self):
        return self.bracketed
    
    def parse_depth(self, i, term=True):
        """Distance to root"""
        if term: # indexing terminal tokens?
            i = self.word2node[self.terminals[i]]
        else:
            i = self.word2node[i]
        
        return self.depth[i]
    
    def path_to_root(self, path, tok):
        """Recursion to will `path` with the ancestors of `tok`"""
        path.append(tok)
        whichedge = np.where(np.array(self.edges)[:,1]==tok)[0]
        if len(whichedge)>0:
            parent = self.edges[whichedge[0]][0]
            path = self.path_to_root(path, parent)
        return path
    
    def tree_dist(self, i, j, term=True):
        if term: # indexing terminal tokens?
            i = self.word2node[self.terminals[i]]
            j = self.word2node[self.terminals[j]]
        else:
            i = self.word2node[i]
            j = self.word2node[j]
            
        # take care of pathological cases
        if i>j:
            i_ = j
            j_ = i
        elif i<j:
            i_ = i
            j_ = j
        else:
            return 0
        if i==0:
            return self.depth[j_]
        
        # get ancestors of both
        anci = np.array(self.path_to_root([], i_)) # these are node indices
        ancj = np.array(self.path_to_root([], j_)) # these are word indices
        
        # get nearest common ancestor
        nearest = np.array(anci)[np.isin(anci,ancj)][0]
        
        # find the parse depth of the nearest common ancestor
        return self.depth[i_] + self.depth[j_] - 2*self.depth[nearest]
    
        
#%%


# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# for i in range(len(labels)):
#     plt.text(ba[i,1], ba[i,2], labels[i], bbox=props)

# plt.xlim([-1,1])
# plt.ylim([-1,1])
# plt.axis('equal')

