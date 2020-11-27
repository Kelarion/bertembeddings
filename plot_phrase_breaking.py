SAVE_DIR = 'C:/Users/mmall/Documents/github/bertembeddings/data/'
CODE_DIR = 'C:/Users/mmall/Documents/github/bertembeddings/'

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
from matplotlib import cm

#%%

order = 1
num_phrases = None
# swap_type = 'within'
swap_type = 'among'

# phrase_type1 = 'real'
# phrase_type2= 'imitation'


# fold = 'shuffle_test/'
# if swap_type == 'among':
#     fold += 'ngrams/'
# # fold = 'ngram_swaps/'
# # if random_model:
# #     fold += 'random_model/'
    
# pref = '%s_%dorder'%(phrase_type1, order)
# # pref = ''
# # if num_phrases is not None:
# #     pref += '_%dphrases'%(num_phrases)
# real_cos = np.load(SAVE_DIR+fold+pref+'_cosines.npy')
# real_frob = np.load(SAVE_DIR+fold+pref+'_frob.npy')
# real_nuc = np.load(SAVE_DIR+fold+pref+'_nuc.npy')
# real_inf = np.load(SAVE_DIR+fold+pref+'_inf.npy')
# real_num_swap = np.load(SAVE_DIR+fold+pref+'_num_swap.npy')
# real_lines = np.load(SAVE_DIR+fold+pref+'_line_id.npy')
# real_num_phrase = np.load(SAVE_DIR+fold+pref+'_num_phrase.npy')

# pref = '%s_%dorder'%(phrase_type2, order)
# # if num_phrases is not None:
# #     pref += '_%dphrases'%(num_phrases)
# fake_cos = np.load(SAVE_DIR+fold+pref+'_cosines.npy')
# fake_frob = np.load(SAVE_DIR+fold+pref+'_frob.npy')
# fake_nuc = np.load(SAVE_DIR+fold+pref+'_nuc.npy')
# fake_inf = np.load(SAVE_DIR+fold+pref+'_inf.npy')
# fake_num_swap = np.load(SAVE_DIR+fold+pref+'_num_swap.npy')
# fake_lines = np.load(SAVE_DIR+fold+pref+'_line_id.npy')
# fake_num_phrase = np.load(SAVE_DIR+fold+pref+'_num_phrase.npy')

fold = 'phrase_swaps/'
if swap_type == 'within':
    fold += 'within/'

cos = np.load(SAVE_DIR+fold+'_cosines.npy')
frob = np.load(SAVE_DIR+fold+'_frob.npy')#/norms.mean(1)[None,:]
# frob_swp = np.load(SAVE_DIR+fold+'_frob_swp.npy')
# frob_unswp = np.load(SAVE_DIR+fold+'_frob_unswp.npy')
nuc = np.load(SAVE_DIR+fold+'_nuc.npy')#/norms.mean(1)[None,:]
inf = np.load(SAVE_DIR+fold+'_inf.npy')#/norms.mean(1)[None,:]
lineid = np.load(SAVE_DIR+fold+'_line_id.npy')
cond = np.load(SAVE_DIR+fold+'_phrase_type.npy')


allcond = np.unique(cond)
froeb_dist = np.array([frob[cond==i,:].mean(0) for i in allcond]).T
nuc_dist = np.array([nuc[cond==i,:].mean(0) for i in allcond]).T
inf_dist = np.array([inf[cond==i,:].mean(0) for i in allcond]).T
cos_sim = np.array([csim[cond==i,:].mean(0) for i in allcond]).T
avg_dist = np.array([distavg[cond==i,:].mean(0) for i in allcond]).T

froeb_dist_err = np.array([frob[cond==i].std(0)/np.sqrt(sum(cond==i)) for i in allcond]).T
nuc_dist_err = np.array([nuc[cond==i].std(0)/np.sqrt(sum(cond==i)) for i in allcond]).T
inf_dist_err = np.array([inf[cond==i].std(0)/np.sqrt(sum(cond==i)) for i in allcond]).T
cos_err = np.array([csim[cond==i,:].std(0)/np.sqrt(sum(cond==i)) for i in allcond]).T
avg_err = np.array([distavg[cond==i,:].std(0)/np.sqrt(sum(cond==i)) for i in allcond]).T

#%%
# cmap = cm.get_cmap('viridis')
cmap = cm.get_cmap('bwr')
mrk = '-.'

col = cmap(allcond/max(allcond))


plot_this = frob
# plot_this = frob_swp
# plot_this = frob_unswp
# plot_this = csim
for i,c in enumerate(allcond):
    mn = plot_this[cond==c,:].mean(0)
    err = plot_this[cond==c,:].std(0)/np.sqrt(np.sum(cond==c))
    
    plt.plot(mn, mrk, marker='o', linewidth=2, color=col[i])
    plt.fill_between(np.arange(13), mn-err, mn+err, alpha=0.4, color=col[i])

plt.legend(allcond, title='n-gram size')
    
# # plot_this1 = real_cos.T
# plot_this1 = real_frob
# # plot_this2 = fake_cos.T
# plot_this2 = fake_frob


# plt.plot(plot_this1.mean(0),marker='o')
# plt.fill_between(np.arange(13),plot_this1.mean(0)-plot_this1.std(0)/np.sqrt(plot_this1.shape[0]), 
#                  plot_this1.mean(0)+plot_this1.std(0)/np.sqrt(plot_this1.shape[0]),
#                  alpha=0.6)


# plt.plot(plot_this2.mean(0),marker='o')
# plt.fill_between(np.arange(13),plot_this2.mean(0)-plot_this2.std(0)/np.sqrt(plot_this2.shape[0]), 
#                  plot_this2.mean(0)+plot_this2.std(0)/np.sqrt(plot_this2.shape[0]),
#                  alpha=0.6)

# plt.legend(['Real phrases', 'Imitation phrase'])
