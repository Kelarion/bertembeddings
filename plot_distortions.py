SAVE_DIR = 'C:/Users/mmall/Documents/github/bertembeddings/data/'
CODE_DIR = 'C:/Users/mmall/Documents/github/bertembeddings/'

import sys, os
sys.path.append(CODE_DIR)
from brackets2trees import BracketedSentence
from spline_regression import spline_covariates

import torch
import tqdm

from transformers import BertTokenizer, BertModel, BertConfig
import pickle as pkl
import numpy as np
import scipy.linalg as la
import scipy.stats as sts
import linecache
from time import time
import matplotlib.pyplot as plt
from matplotlib import cm

# %%
# random_model = True
random_model = False

use_subwords = False
# use_subwords = True

# exp_type = 'ngram size'
# exp_type = 'phrase'
# exp_type = 'phrase type'
exp_type = 'tree dist'

jonathan_data = False
# jonathan_data = True

if exp_type == 'ngram size':
    fold = 'ngram_swaps/'
elif exp_type == 'phrase':
    fold = 'phrase_swaps/'
elif exp_type == 'phrase type':
    fold = 'phrase_swaps/vpnp/'
elif exp_type == 'tree dist':
    fold = 'bracket_crossings/full_features/'
    
if jonathan_data:
    fold += 'jonathans/'

if use_subwords:
    fold += 'using_subwords/'
if random_model:
    fold += 'random_model/'

# norms = np.load(SAVE_DIR+fold+'_average_norms.npy')
cos = np.load(SAVE_DIR+fold+'_cosines.npy')
frob = np.load(SAVE_DIR+fold+'_frob.npy')#/norms.mean(1)[None,:]
# frob_swp = np.load(SAVE_DIR+fold+'_frob_swp.npy')
# frob_unswp = np.load(SAVE_DIR+fold+'_frob_unswp.npy')
nuc = np.load(SAVE_DIR+fold+'_nuc.npy')#/norms.mean(1)[None,:]
inf = np.load(SAVE_DIR+fold+'_inf.npy')#/norms.mean(1)[None,:]
lineid = np.load(SAVE_DIR+fold+'_line_id.npy')
cond = np.load(SAVE_DIR+fold+'_condition.npy')
# cond = np.load(SAVE_DIR+fold+'_phrase_type.npy')
# swapid = np.load(SAVE_DIR+fold+'_swap_id.npy')
# distavg = np.load(SAVE_DIR+fold+'_dist_avg.npy')#/norms.mean(1)[None,:]
# frob = norms

if (exp_type != 'tree dist') and not jonathan_data:
    num_swap = np.load(SAVE_DIR+fold+'_num_swap.npy')
    dist_swap = np.load(SAVE_DIR+fold+'_dist_swap.npy')
    num_tok = np.load(SAVE_DIR+fold+'_num_tok.npy')

# csim = np.array([cos[:,swapid==s].mean(1) for s in np.unique(swapid)])

allcond = np.unique(cond)
froeb_dist = np.array([frob[cond==i,:].mean(0) for i in allcond]).T
nuc_dist = np.array([nuc[cond==i,:].mean(0) for i in allcond]).T
inf_dist = np.array([inf[cond==i,:].mean(0) for i in allcond]).T
# cos_sim = np.array([csim[cond==i,:].mean(0) for i in allcond]).T
# avg_dist = np.array([distavg[cond==i,:].mean(0) for i in allcond]).T

froeb_dist_err = np.array([frob[cond==i].std(0)/np.sqrt(sum(cond==i)) for i in allcond]).T
nuc_dist_err = np.array([nuc[cond==i].std(0)/np.sqrt(sum(cond==i)) for i in allcond]).T
inf_dist_err = np.array([inf[cond==i].std(0)/np.sqrt(sum(cond==i)) for i in allcond]).T
# cos_err = np.array([csim[cond==i,:].std(0)/np.sqrt(sum(cond==i)) for i in allcond]).T
# avg_err = np.array([distavg[cond==i,:].std(0)/np.sqrt(sum(cond==i)) for i in allcond]).T

#%%
cmap = cm.get_cmap('viridis')
# cmap = cm.get_cmap('bwr')
mrk = '-'

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

plt.legend(allcond, title=exp_type)

#%%
# control_for = dist_swap/num_tok
control_for = num_swap/num_tok

# cmap = cm.get_cmap('viridis')
cmap = cm.get_cmap('bwr')
if random_model:
    mrk = '--'
else:
    mrk = '-'

col = cmap([0.,1.])

regs = np.array([spline_covariates((cond), control_for, (frob[:,l]), 
                                   compute_before=True, num_basis=num_bins, sandwich=True) \
                 for l in range(13)])

conf = 2*regs[:,:,2]

# plt.figure()
# for i in [0,1]:
plt.plot(regs[:,1,0],mrk,color='k',linewidth=2)
plt.fill_between(np.arange(13), regs[:,1,0]-conf[:,1], regs[:,1,0]+conf[:,1], alpha=0.4, color='k')

plt.title(r'Distortion ~ %s | swap size '%exp_type)
plt.ylabel('Coefficient (95% conf. interval)')
plt.xlabel('Layer')
# plt.legend(['Naive','Controlled'],title='Regression type')

# plt.plot(plt.xlim(),[0,0],'k')

#%% Tree-distance specific: permutation test



regs = np.array([spline_covariates((cond), np.random.randn(len(cond)), (frob[:,l]), 
                                   compute_before=True, num_basis=num_bins, sandwich=True) \
                 for l in range(13)])

conf = 2*regs[:,:,2]

# plt.figure()
# for i in [0,1]:
plt.plot(regs[:,0,0],mrk,color='k',linewidth=2)
plt.fill_between(np.arange(13), regs[:,0,0]-conf[:,0], regs[:,0,0]+conf[:,0], alpha=0.4, color='k')

plt.title(r'Distortion ~ %s | swap size '%exp_type)
plt.ylabel('Coefficient (95% conf. interval)')
plt.xlabel('Layer')
# plt.legend(['Naive','Controlled'],title='Regression type')




