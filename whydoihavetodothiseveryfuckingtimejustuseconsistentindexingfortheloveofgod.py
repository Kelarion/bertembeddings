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
jon_folder = 'C:/Users/mmall/Documents/github/bertembeddings/data/jonathans/'

lines = os.listdir(jon_folder+model)
dist = pkl.load(open(jon_folder+'/permuted_data.pkl','rb'))

dfile = SAVE_DIR+'train_bracketed.txt'

#%%

ptb_in_perm = []
for line_idx in range(2416):
    
    line_ptb = linecache.getline(dfile, line_idx+1)
    sent = BracketedSentence(line_ptb).words
    
    found = False
    for pd_line in np.random.permutation(range(len(dist))):
        if sent == dist[pd_line][0]:
            ptb_in_perm.append(pd_line)
            print('Found one')
            break
    