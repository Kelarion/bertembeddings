SAVE_DIR = '/home/matteo/Documents/github/bertembeddings/data/'
CODE_DIR = '/home/matteo/Documents/github/bertembeddings/'

import sys
sys.path.append(CODE_DIR)
from brackets2trees import BracketedSentence

import torch
import tqdm

from transformers import BertTokenizer, BertModel, BertConfig
import pickle as pkl
import numpy as np
import scipy.linalg as la
import linecache
from time import time
import matplotlib.pyplot as plt

#%%

order = 2
num_phrases = 2
# swap_type = 'within'
swap_type = 'among'

phrase_type1 = 'real'
phrase_type2 = 'imitation'


fold = 'shuffle_test/'
if swap_type == 'among':
    fold += 'ngrams/'
    
pref = '%s_%dorder'%(phrase_type1, order)
if num_phrases is not None:
    pref += '_%dphrases'%(num_phrases)
real_cos = np.load(SAVE_DIR+fold+pref+'_cosines.npy')
real_frob = np.load(SAVE_DIR+fold+pref+'_frob.npy')
real_nuc = np.load(SAVE_DIR+fold+pref+'_nuc.npy')
real_inf = np.load(SAVE_DIR+fold+pref+'_inf.npy')
real_num_swap = np.load(SAVE_DIR+fold+pref+'_num_swap.npy')
real_lines = np.load(SAVE_DIR+fold+pref+'_line_id.npy')
real_num_phrase = np.load(SAVE_DIR+fold+pref+'_num_phrase.npy')

pref = '%s_%dorder'%(phrase_type2, order)
if num_phrases is not None:
    pref += '_%dphrases'%(num_phrases)
fake_cos = np.load(SAVE_DIR+fold+pref+'_cosines.npy')
fake_frob = np.load(SAVE_DIR+fold+pref+'_frob.npy')
fake_nuc = np.load(SAVE_DIR+fold+pref+'_nuc.npy')
fake_inf = np.load(SAVE_DIR+fold+pref+'_inf.npy')
fake_num_swap = np.load(SAVE_DIR+fold+pref+'_num_swap.npy')
fake_lines = np.load(SAVE_DIR+fold+pref+'_line_id.npy')
fake_num_phrase = np.load(SAVE_DIR+fold+pref+'_num_phrase.npy')


#%%
    
plot_this1 = real_cos.T
# plot_this1 = real_frob
plot_this2 = fake_cos.T
# plot_this2 = fake_frob


plt.plot(plot_this1.mean(0),marker='o')
plt.fill_between(np.arange(13),plot_this1.mean(0)-plot_this1.std(0)/np.sqrt(plot_this1.shape[0]), 
                 plot_this1.mean(0)+plot_this1.std(0)/np.sqrt(plot_this1.shape[0]),
                 alpha=0.6)


plt.plot(plot_this2.mean(0),marker='o')
plt.fill_between(np.arange(13),plot_this2.mean(0)-plot_this2.std(0)/np.sqrt(plot_this2.shape[0]), 
                 plot_this2.mean(0)+plot_this2.std(0)/np.sqrt(plot_this2.shape[0]),
                 alpha=0.6)

plt.legend(['Real phrases', 'Imitation phrase'])
