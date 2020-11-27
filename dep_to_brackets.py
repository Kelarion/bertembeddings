import numpy as np
import pandas
import linecache

CODE_DIR = '/home/matteo/Documents/github/bertembeddings/'
SAVE_DIR = '/home/matteo/Documents/github/bertembeddings/data/bert/'
LOAD_DIR = '/home/matteo/Documents/github/bertembeddings/data/bert/'

import sys, os
# sys.path.append(CODE_DIR+'repler/src/')

sys.path.append(CODE_DIR)
from brackets2trees import BracketedSentence

#%%
dep = open(LOAD_DIR+'/dependency_train.txt')

tok_num = dep[0].values
line_num = np.cumsum(np.diff(tok_num, prepend=0)<0)
tokens = dep[1].values
pos = dep[7].values
parents = dep[6].values
sent_length = np.unique(line_num, return_counts=True)[1]

idx = np.load(LOAD_DIR+'dep_idx_in_const.npy')
#%%
def add_children(k, sent, l):
    """recursion to find and add children of token k"""
    word = tokens[line_num==l][tok_num[line_num==l]==k][0]
    tag = pos[line_num==l][tok_num[line_num==l]==k][0]
    
    sent += ' (%s %s %d'%(tag, word, k-1)
    
    children = list(tok_num[line_num==l][parents[line_num==l]==k])
    for j in children:
        sent = add_children(j, sent, l)
    sent += ')'
    
    return sent

these_lines = np.where(~np.isnan(idx))[0]
with open(SAVE_DIR+'dependency_train_bracketed.txt','w') as bs_file:
    for ln,l in enumerate(these_lines):
        root_id = tok_num[line_num==l][parents[line_num==l]==0][0]
        
        sent = ''
        bracketed = add_children(root_id, sent, l)
        
        bs_file.write(bracketed+'\n')
        
        print('Saved line %d/%d'%(ln,len(these_lines)))


