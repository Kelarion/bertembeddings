SAVE_DIR = 'C:/Users/mmall/Documents/github/bertembeddings/data/'
CODE_DIR = 'C:/Users/mmall/Documents/github/bertembeddings/'

import sys, os
sys.path.append(CODE_DIR)
from brackets2trees import BracketedSentence

import torch
import tqdm

from transformers import BertTokenizer, BertModel, BertConfig, AutoConfig, AutoModel, AutoTokenizer
import pickle as pkl
import numpy as np
import scipy.linalg as la
import linecache
from time import time
import matplotlib.pyplot as plt

#%%
trn = open(SAVE_DIR+'/train_bracketed.txt','r')
tst = open(SAVE_DIR+'/test_bracketed.txt','r')

pos_tags = []
phrase_tags = []
for line in tqdm.tqdm(trn):
    sentence = BracketedSentence(line)
    pos_tags = list(set(pos_tags + sentence.pos_tags))
    # all_tags = list(set(all_tags + sentence.node_tags))
    phr = list(np.array(sentence.node_tags)[np.setdiff1d(sentence.word2node,sentence.term2word)])
    # phr = [sentence.ancestor_tags(i, 2) for i in range(sentence.ntok)]
    phrase_tags = list(set(phrase_tags + phr))

for line in tqdm.tqdm(tst):
    sentence = BracketedSentence(line)
    pos_tags = list(set(pos_tags + sentence.pos_tags))
    # all_tags = list(set(all_tags + sentence.node_tags))
    phr = list(np.array(sentence.node_tags)[np.setdiff1d(sentence.word2node,sentence.term2word)])
    # phr = [sentence.ancestor_tags(i, 2) for i in range(sentence.ntok)]
    phrase_tags = list(set(phrase_tags + phr))
    
#%%
trn = open(SAVE_DIR+'/train_bracketed.txt','r')
tst = open(SAVE_DIR+'/test_bracketed.txt','r')

pos_tag_idx = []
phrase_tag_idx = []

for line in tqdm.tqdm(trn):
    sentence = BracketedSentence(line)
    pos_tag_idx += [pos_tags.index(w) for w in sentence.pos_tags]
    
    phr = list(np.array(sentence.node_tags)[np.setdiff1d(sentence.word2node,sentence.term2word)])
    # phr = [sentence.ancestor_tags(i, 2) for i in range(sentence.ntok)]
    phrase_tag_idx += [phrase_tags.index(w) for w in phr]

for line in tqdm.tqdm(tst):
    sentence = BracketedSentence(line)
    pos_tag_idx += [pos_tags.index(w) for w in sentence.pos_tags]
    
    phr = list(np.array(sentence.node_tags)[np.setdiff1d(sentence.word2node,sentence.term2word)])
    # phr = [sentence.ancestor_tags(i, 2) for i in range(sentence.ntok)]
    phrase_tag_idx += [phrase_tags.index(w) for w in phr]

#%%
unq_pos, cnt_pos = np.unique(pos_tag_idx, return_counts=True)
good_pos = (cnt_pos/len(pos_tag_idx))>1e-3
final_pos_tags = list(np.array(pos_tags)[good_pos])

unq_phrase, cnt_phrase = np.unique(phrase_tag_idx, return_counts=True)
good_phrs = (cnt_phrase/len(phrase_tag_idx))>1e-3
final_phrase_tags = list(np.array(phrase_tags)[good_phrs])

np.save(SAVE_DIR+'unique_pos_tags.npy', final_pos_tags)
np.save(SAVE_DIR+'pos_tag_frequency.npy', cnt_pos[good_pos]/np.sum(cnt_pos[good_pos]))
np.save(SAVE_DIR+'unique_phrase_tags.npy', final_phrase_tags)
np.save(SAVE_DIR+'phrase_tag_frequency.npy', cnt_phrase[good_phrs]/np.sum(cnt_phrase[good_phrs]))


