SAVE_DIR = '/home/matteo/Documents/github/bertembeddings/data/'
CODE_DIR = '/home/matteo/Documents/github/bertembeddings/'

import sys, os
sys.path.append(CODE_DIR)
from brackets2trees import BracketedSentence

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
random_model = True

# fold = 'bracket_crossings/full_features/'
fold = 'ngram_swaps/'
if random_model:
    fold += 'random_model/'

norms = np.load(SAVE_DIR+fold+'_average_norms.npy')
cos = np.load(SAVE_DIR+fold+'_cosines.npy')
frob = np.load(SAVE_DIR+fold+'_frob.npy')/norms#.mean(1)[None,:]
nuc = np.load(SAVE_DIR+fold+'_nuc.npy')/norms#.mean(1)[None,:]
inf = np.load(SAVE_DIR+fold+'_inf.npy')/norms#.mean(1)[None,:]
lineid = np.load(SAVE_DIR+fold+'_line_id.npy')
cond = np.load(SAVE_DIR+fold+'_num_crossings.npy')
swapid = np.load(SAVE_DIR+fold+'_swap_id.npy')
distavg = np.load(SAVE_DIR+fold+'_dist_avg.npy')/norms#.mean(1)[None,:]

csim = np.array([cos[:,swapid==s].mean(1) for s in np.unique(swapid)])

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

plot_this = frob
# plot_this = csim
for c in allcond:
    mn = plot_this[cond==c,:].mean(0)
    err = plot_this[cond==c,:].std(0)/np.sqrt(np.sum(cond==c))
    
    plt.plot(mn, marker='o')
    plt.fill_between(np.arange(13), mn-err, mn+err, alpha=0.6)

plt.legend(allcond, title='n-gram size')

#%%
whichlayer = 8

plot_this = froeb_dist
error = froeb_dist_err
plt.subplot(2,3,1)
for t in range(len(allcond)):
    plt.plot(np.arange(13),plot_this[:,t], marker='o')
    plt.fill_between(np.arange(13),   
                     plot_this[:,t]-error[:,t], 
                     plot_this[:,t]+error[:,t],
                     alpha=0.5)

plt.legend(allcond+2, title='Tree distance')
plt.ylabel('$||Original - Swapped||_F$')
plt.gca().set_xticklabels(['LAYER%d'%l for l in range(13)])

plot_this = avg_dist
error = avg_err
plt.subplot(2,3,2)
for t in range(len(allcond)):
    plt.plot(np.arange(1,14),plot_this[:,t], marker='o')
    plt.fill_between(np.arange(1,14), 
                     plot_this[:,t]-error[:,t], 
                     plot_this[:,t]+error[:,t],
                     alpha=0.5)

plt.legend(allcond+2, title='num crossings')
plt.ylabel('$||Original - Swapped||_nuc$')

plot_this = inf_dist
error = inf_dist_err
plt.subplot(2,3,3)
for t in range(len(allcond)):
    plt.plot(np.arange(1,14),plot_this[:,t], marker='o')
    plt.fill_between(np.arange(1,14), 
                     plot_this[:,t]-error[:,t], 
                     plot_this[:,t]+error[:,t],
                     alpha=0.5)

plt.legend(allcond, title='num crossings')
plt.ylabel('$||Original - Swapped||_{\infty}$')

# scatter plots

plt.subplot(2,3,4)
plot_this = frob
rho = [sts.spearmanr(plot_this[:,l], cond)[0] for l in range(13)]
pval = [sts.spearmanr(plot_this[:,l], cond)[1] for l in range(13)]
plt.scatter(cond+2  +np.random.randn(len(cond))*0.1,
            plot_this[:,whichlayer])
plt.xlabel('Tree distance')
plt.ylabel('$||Original - Swapped||_F$')
plt.title('Layer %d: spr %.3f, p=%.3f'%(whichlayer+1, rho[whichlayer], pval[whichlayer]))

plt.subplot(2,3,5)
plot_this = distavg
rho = [sts.spearmanr(plot_this[:,l], cond)[0] for l in range(13)]
pval = [sts.spearmanr(plot_this[:,l], cond)[1] for l in range(13)]
plt.scatter(cond+np.random.randn(len(cond))*0.1,
            plot_this[:,whichlayer])
plt.xlabel('number of crossings')
plt.ylabel('$||Original - Swapped||_nuc$')
plt.title('Layer %d: spr %.3f, p=%.3f'%(whichlayer+1, rho[whichlayer], pval[whichlayer]))


plt.subplot(2,3,6)
plot_this = inf
rho = [sts.spearmanr(plot_this[:,l], cond)[0] for l in range(13)]
pval = [sts.spearmanr(plot_this[:,l], cond)[1] for l in range(13)]
plt.scatter(cond+np.random.randn(len(cond))*0.1,
            plot_this[:,whichlayer])
plt.xlabel('number of crossings')
plt.ylabel('$||Original - Swapped||_{\infty}$')
plt.title('Layer %d: spr %.3f, p=%.3f'%(whichlayer+1, rho[whichlayer], pval[whichlayer]))
