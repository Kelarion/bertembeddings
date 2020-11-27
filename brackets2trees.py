"""
A class for interacting with a parsed sentence.

__init__ inputs:
    bs: the bracketed sentence string
    dep_tree (Bool, default False): is this a dependency parse?

Attributes (most of them):
    edges: list of tuples, (parent, child), representing the edges of the tree
    nodes: list of strings, 'POS [word]', of all tokens (terminal and non-terminal)
    words: list of strings, 'word', which are the words of the sentence (i.e. terminals)
    brackets: array of {+1, -1, 0}, for {opening, closing, terminal tokens}
    bracketed_string: string, the input used to create the object
    parents: array, the parent of each token (is -1 for the root)
    node_span: array, for each token, the index of the token right after its constituent ends
    subtree_order: array, ints, the `order' of each token

Note that attributes which return indices, they index the tokens in the order
they appear in the bracketed sequence -- for a dependency parse, this doesn't 
generally match the order in the actualy sentence. It also doesn't match the 
indices of word in the sentence (which go from 0, ..., num_terminal_tokens).
To convert between these three indexing coordinates -- the 

Methods (external use):
    tree_dist(i, j, term=True):
        Distance between tokens i and j on the parrse tree, returns an integer
        `term` toggles whether i and j index the words, or all nodes
    bracket_dist(i, j):
        Number of (open or closed) brackets between i and j, not including 
        brackets associated with terminal tokens.
        Only works on terminal tokens
    is_relative(i, j, order=1):
        Are tokens i and j in the same phrase?
    
"""

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
        Take a bracketed sentence, bs, and compute various quantities defined 
        on it and/or the corresponding parse tree. Provides methods to ask
        questions about the tokens and subtrees.
        
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
        term_cl = braks[np.diff(iscl.astype(int),append=0)==-1]
        
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
        self.term2word = leaves # indices of terminal tokens in the sentence
        
        parents = [edges[np.where(np.array(edges)[:,1]==i)[0][0]][0] \
                   for i in range(1,len(nodes))]
        self.parents = np.append(-1,parents) # seq-indexed
        
        self.depth = depth
        
        # Find the order of each subtree 
        depf = np.append(self.depth,0)
        # first, for each node find the node right after its constituent ends
        span = [(depf[i+1:]<=depf[i]).argmax()+(i+1) for i in range(len(self.nodes))]
        # then find the maxmimum distance between each node and its descendants
        max_depth = [self.depth[i:span[i]].max()-self.depth[i] for i in range(len(self.nodes))]
        self.node_span = np.array(span) # each token's corresponding closing
        self.subtree_order = np.array(max_depth) # each node's order (=0 for terminals)
        
        # Represent the bracketed sentence as +/- 1 string
        # brah = np.sort(np.append(op,cl))
        # brah[np.isin(braks,op)] = 1
        # brah[np.isin(braks,cl)] = -1
        # brah[np.diff(iscl.astype(int),append=0)==1] = 0
        # # brah[np.diff(iscl.astype(int),append=0)==-1] = 0
        brah = np.array([1 if b in op[~np.isin(op,term_op)] \
                         else -1 if b in cl[~np.isin(cl,term_cl)] \
                             else 0 if b in term_op \
                                 else np.nan for b in braks])
        brah = brah[~np.isnan(brah)]
        self.brackets = brah.astype(int)
        self.term2brak = np.where(brah==0)[0]
        
        # # old code, not sure what exactly it computes
        # is_trough = np.flip(np.diff(np.flip(~iscl*2-1),prepend=1)) == -2
        # trough_depths = np.cumsum(~iscl*2-1)[is_trough]-1
        # is_trough[-1] = False
        # subtree_depths = trough_depths[np.cumsum(is_trough)][np.isin(braks,op)]
        # self.subtree_depth = subtree_depths
        
        # 
        if dep_tree:
            line = [n.split(' ')[1] for n in labels]
        else:
            words = np.array(labels)[leaves]
            line = [w.split(' ' )[1] for w in words]
        self.words = line
        self.ntok = len(line)
        
        self.bracketed_string = bs
        
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
            
        self.node2word = np.array(node_names) # indices of each node in the sequence
        self.word2node = np.array(word_names) # indices of each word in the node list
        self.node2term = np.array([leaves.tolist().index(i) if i in leaves else -1 \
                                   for i in node_names])
        
        self.node_tags = [w.split(' ')[0] for w in self.nodes]
        self.pos_tags = list(np.array(self.node_tags)[leaves])
        
    def __repr__(self):
        return self.bracketed_string
    
    def parse_depth(self, i, term=True):
        """Distance to root"""
        if term: # indexing terminal tokens?
            i = self.word2node[self.term2word[i]]
        else:
            i = self.word2node[i]
        
        return self.depth[i]
    
    def path_to_root(self, path, tok):
        """Recursion to fill `path` with the ancestors of `tok`"""
        path.append(tok)
        whichedge = np.where(np.array(self.edges)[:,1]==tok)[0]
        if len(whichedge)>0:
            parent = self.edges[whichedge[0]][0]
            path = self.path_to_root(path, parent)
        return path
    
    def tree_dist(self, i, j, term=True):
        """
        Compute d = depth(i) + depth(j) - 2*depth(nearest common ancestor)
        """
        if term: # indexing terminal tokens?
            i = self.word2node[self.term2word[i]]
            j = self.word2node[self.term2word[j]]
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
        anci = np.array(self.path_to_root([], i_))  # [i, parent(i), ..., 0]
        ancj = np.array(self.path_to_root([], j_))  # [j, parent(j), ..., 0]
        
        # get nearest common ancestor
        nearest = np.array(anci)[np.isin(anci,ancj)][0]
        
        return self.depth[i_] + self.depth[j_] - 2*self.depth[nearest]
    
    def ancestor_tags(self, i, n=2):
        i_ = self.word2node[self.term2word[i]]
        anc = np.array(self.path_to_root([], i_)) # all ancestors
        return self.node_tags[anc[np.min([n, len(anc)-1])]] # choose the nth one
    
    def phrases(self, order=1, min_length=2, strict=False):
        """
        Phrases of a given order are subtrees whose deepest member is at 
        most `order' away from the subtree root
        
        Returns a list of arrays, which contain the indices of all phrases of 
        specified order (if strict) or at least specified order (if not strict)
        
        Note that is strict=False, some indices might appear twice, as phrases 
        of lower order are nested in phrases of higher order.
        """
        if strict:
            phrs = np.where((self.subtree_order!=0)&(self.subtree_order==order))[0]
        else:
            phrs = np.where((self.subtree_order!=0)&(self.subtree_order<=order))[0]
        
        phrases = [np.array(range(i+1,self.node_span[i])) for i in phrs]
        chunks = [self.node2term[p[np.isin(p, self.term2word)]] for p in phrases]
        
        return [c for c in chunks if len(c)>=min_length]
    
    def bracket_dist(self,i,j):
        """Number of (non-terminal) brackets between i and j. Only guaranteed 
        to be meaningful for adjacent terminal tokens."""
        i = self.node2term[self.word2node[self.term2word[i]]]
        j = self.node2term[self.word2node[self.term2word[j]]]
            
        return np.abs(self.brackets)[self.term2brak[i]:self.term2brak[j]+1].sum()
        
    def is_relative(self, i, j, order=1, term=True):
        """
        Bool, are tokens i and j part of the same n-th order subtree?
        Equivalently: is the nearest common ancestor of i and j of the specified
        order?
        """
        
        if term: # indexing terminal tokens?
            i = self.word2node[self.term2word[i]]
            j = self.word2node[self.term2word[j]]
        else:
            i = self.word2node[i]
            j = self.word2node[j]
        
        if order==1 and (self.depth[i]!=self.depth[j]):
            return False # a necessary condition
        
        anci = np.array(self.path_to_root([], i)) # these are node indices
        ancj = np.array(self.path_to_root([], j)) # these are word indices
        nearest = np.array(anci)[np.isin(anci,ancj)][0]
        
        return self.subtree_order[nearest]<=order

