SAVE_DIR = '/home/matteo/Documents/github/bertembeddings/data/'
CODE_DIR = '/home/matteo/Documents/github/bertembeddings/'

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

#%%
fold = 'bracket_crossings/'

csim = np.load(SAVE_DIR+fold+'_cosines.npy')
frob = np.load(SAVE_DIR+fold+'_frob.npy')
nuc = np.load(SAVE_DIR+fold+'_nuc.npy')
inf = np.load(SAVE_DIR+fold+'_inf.npy')
lineid = np.load(SAVE_DIR+fold+'_line_id.npy')
cond = np.load(SAVE_DIR+fold+'_num_crossings.npy')
swapid = np.load(SAVE_DIR+fold+'_swap_id.npy')
attn = np.load(SAVE_DIR+fold+'_attn.npy')
attn_orig = np.load(SAVE_DIR+fold+'_attn_orig.npy')

# csim = np.array([cos[swapid==s,:,:].mean(2) for s in np.unique(swapid)])
distavg = np.load(SAVE_DIR+fold+'_dist_avg.npy')

#%%
# plot_this = attn
plot_this = frob

for l in range(12):
    for h in range(12):
        n = 12*h + l + 1
        plt.subplot(12,12,n)
        mean = np.array([plot_this[cond==c,h,l].mean() for c in np.unique(cond)])
        var = np.array([plot_this[cond==c,h,l].std() for c in np.unique(cond)])
        plt.errorbar(np.unique(cond), mean, yerr = var)
        # plt.hist(attn_orig[cond==0,h,l], alpha=0.6, density=True)
        # plt.hist(attn_orig[cond==2,h,l], alpha=0.6, density=True)
        if l==0:
            plt.ylabel('attn')
        else:
            plt.yticks([])
        if h==11:
            plt.xlabel('dist')
        else:
            plt.xticks([])

#%%
# test_this = 1-csim
test_this = frob

rho = np.array([[sts.spearmanr(cond, test_this[:,i,j])[0] for j in range(12)] for i in range(12)])
pval = np.array([[sts.spearmanr(cond, test_this[:,i,j])[1] for j in range(12)] for i in range(12)])

rho_attn = np.array([[sts.spearmanr(cond, attn[:,i,j])[0] for j in range(12)] for i in range(12)])
pval_attn = np.array([[sts.spearmanr(cond, attn[:,i,j])[1] for j in range(12)] for i in range(12)])

layers = np.arange(12)[:,None]*np.ones(12)[None,:]

sct = plt.scatter(rho_attn.flatten(), rho.flatten(), 
                  c=layers.flatten(),
                  s=20*(1-pval.flatten())+5)
cb = plt.colorbar(sct,ticks=range(1,13), drawedges=True, values=range(1,13))
plt.axis('scaled')
plt.xlim([np.min(plt.xlim()+plt.ylim()),np.max(plt.xlim()+plt.ylim())])
plt.ylim(plt.xlim())

plt.plot(plt.xlim(),plt.xlim(),'--',c=(0.5,0.5,0.5))
plt.xlabel('rank corr (attention, tree distance)')
plt.ylabel('rank corr (distortion, tree distance)')

#%%
alone = np.zeros((12,12,2))
with_attn = np.zeros((12,12,2))
for l in range(12):
    for h in range(12):
        wout, w = spline_covariates(cond, attn[:,h,l], test_this[:,h,l], compute_before=True)
        alone[l,h,:] = wout
        with_attn[l,h,:] = w

#%%
ax = plt.axes()
sct = ax.scatter(np.log10(alone[:,:,1].flatten()+1e-10), np.log10(with_attn[:,:,1].flatten()+1e-10), 
                 c=rho.flatten(), cmap='coolwarm', vmin=-0.49, vmax=0.49)
plt.axis('equal')
plt.axis('square')

cb = plt.colorbar(sct)
# cb.set_label('absolute rank-corr(distortion, tree distance)')

plt.plot(plt.xlim(),plt.xlim(), '--', c=(0.5,0.5,0.5))

labs = list(ax.get_xticklabels())
text = [b.get_text() for b in labs]
ax.set_yticklabels(['$10^{%s}$'%l for l in text])
ax.set_xticklabels(['$10^{%s}$'%l for l in text])

plt.xlabel('p-value without attention as covariate', fontsize=13)
plt.ylabel('p-value with attention as covariate', fontsize=13)

