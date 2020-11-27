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

#%%
def diverging_clim(ax):
    for im in ax.get_images():
        cmax = np.max(np.abs(im.get_clim()))
        im.set_clim(-cmax,cmax)

#%%
# random_model = True
random_model = False

fold = 'bracket_crossings/'

if random_model:
    fold += 'random_model/'

csim = np.load(SAVE_DIR+fold+'_cosines.npy')
frob = np.load(SAVE_DIR+fold+'_frob.npy')
# frob_full = np.load(SAVE_DIR+fold+'_frob_full.npy')
nuc = np.load(SAVE_DIR+fold+'_nuc.npy')
inf = np.load(SAVE_DIR+fold+'_inf.npy')
lineid = np.load(SAVE_DIR+fold+'_line_id.npy')
cond = np.load(SAVE_DIR+fold+'_num_crossings.npy')
swapid = np.load(SAVE_DIR+fold+'_swap_id.npy')
# attn = np.load(SAVE_DIR+fold+'_attn_swap.npy')
attn_orig = np.load(SAVE_DIR+fold+'_attn_orig.npy')
# conc = np.load(SAVE_DIR+fold+'_concentration.npy')

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

#%% Spearman Correlation
# test_this = 1-csim
test_this = frob

which_attn = attn
# which_attn = attn_orig
# which_attn = attn-attn_orig

rho = np.array([[sts.spearmanr(cond, test_this[:,i,j])[0] for j in range(12)] for i in range(12)])
pval = np.array([[sts.spearmanr(cond, test_this[:,i,j])[1] for j in range(12)] for i in range(12)])

rho_attn = np.array([[sts.spearmanr(cond, which_attn[:,i,j])[0] for j in range(12)] for i in range(12)])
pval_attn = np.array([[sts.spearmanr(cond, which_attn[:,i,j])[1] for j in range(12)] for i in range(12)])

layers = (np.arange(12)[:,None]*np.ones(12)[None,:])

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

#%% plot sorted
idx = np.argsort((attn-attn_orig).mean(0),1)
# idx = np.argsort(rho_attn,1)

# plot_this = (attn-attn_orig).mean(0)
# plot_this = (1/conc).mean(0)
plot_this = rho_attn
# plot

plt.figure()
plt.imshow(np.take_along_axis(plot_this,idx, 1),extent=[0.5,12.5,0.5,12.5],cmap='bwr')
diverging_clim(plt.gca())


#%%
n_perms = 100

for l in range(12):
    for h in range(12):
        n = 12*h + l + 1
        plt.subplot(12,12,n)
        
        nulls = []
        for n in range(n_perms):
            nulls.append(sts.spearmanr(np.random.permutation(cond), test_this[:,l,h])[0])
        
        plt.hist(nulls, density=True, alpha=0.6)
        # plt.errorbar(np.unique(cond), mean, yerr = var)
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


#%% Spline covariate regression
# which_attn = attn
# which_attn = attn_orig
which_attn = attn-attn_orig
# which_attn = 0.5*(attn+attn_orig)

# test_this = 1-csim
test_this = frob

alone = np.zeros((12,12,3))
with_attn = np.zeros((12,12,3))
for l in range(12):
    for h in range(12):
        wout, w = spline_covariates(sts.rankdata(cond), which_attn[:,l,h], sts.rankdata(test_this[:,l,h]), 
                                    compute_before=True, sandwich=True)
        alone[l,h,:] = wout
        with_attn[l,h,:] = w

#%% Plot spline results
cmap = cm.get_cmap('viridis')
# cmap = cm.get_cmap('bwr')
colorby = layers.flatten()

col = cmap(colorby/max(colorby))

layers = (np.arange(12)[:,None]*np.ones(12)[None,:])

ax = plt.axes()
# sct = ax.scatter(np.log10(alone[:,:,1].flatten()+1e-10), np.log10(with_attn[:,:,1].flatten()+1e-10), 
#                   c=rho_attn.flatten(), cmap='coolwarm', vmin=-0.2, vmax=0.2)
sct = ax.scatter((alone[:,:,0].flatten()), (with_attn[:,:,0].flatten()), 
                  c=col, zorder=10)

errs = ax.errorbar(alone[:,:,0].flatten(), with_attn[:,:,0].flatten(),
                    2*alone[:,:,2].flatten(), 2*with_attn[:,:,2].flatten(),
                    elinewidth=1,linewidth=0, ecolor=col, zorder=0)

newlims = [np.min([plt.ylim(), plt.xlim()]), np.max([plt.ylim(), plt.xlim()])]

plt.axis('equal')
plt.axis('square')
plt.xlim(newlims)
plt.ylim(newlims)

# cb = plt.colorbar(sct)
cb = plt.colorbar(sct,ticks=range(1,13), drawedges=True, values=range(1,13), label='Layer')
# cb.set_label('absolute rank-corr(distortion, tree distance)')

plt.plot(newlims,newlims, '--', c=(0.5,0.5,0.5))
plt.plot(newlims,[0,0], '--', c=(0.5,0.5,0.5))
plt.plot([0,0],newlims, '--', c=(0.5,0.5,0.5))

# labs = list(ax.get_xticklabels())
# text = [b.get_text() for b in labs]
# ax.set_yticklabels(['$10^{%s}$'%l for l in text[1:]])
# ax.set_xticklabels(['$10^{%s}$'%l for l in text[1:]])

# plt.xlabel('p-value without attention as covariate', fontsize=13)
# plt.ylabel('p-value with attention as covariate', fontsize=13)
plt.xlabel(r'$\rho (distortion, tree \, dist)$', fontsize=13)
plt.ylabel(r'$\rho (distortion, tree \, dist \vert \Delta a_{swp})$', fontsize=13)

#%%

errs = plt.errorbar(rho_unt.flatten(), rho.flatten(),
                   np.abs(np.stack([conf_low_unt.flatten(),conf_up_unt.flatten()])-rho_unt.flatten()),
                   np.abs(np.stack([conf_low.flatten(),conf_up.flatten()])-rho.flatten()),
                    elinewidth=1,linewidth=0, ecolor=col, zorder=0)

newlims = [np.min([plt.ylim(), plt.xlim()]), np.max([plt.ylim(), plt.xlim()])]

plt.axis('equal')
plt.axis('square')
plt.xlim(newlims)
plt.ylim(newlims)


#%% Attention ratio figure
atrat = np.array([[attn[cond==0,i,j].mean()/attn[cond>0,i,j].mean() for i in range(12)] for j in range(12)])
drat = np.array([[frob[cond==0,i,j].mean()/frob[cond>0,i,j].mean() for i in range(12)] for j in range(12)])

L = np.arange(12)[None,:]*np.ones((12,1))

plt.scatter(atrat.flatten(),drat.flatten(), c=L.flatten(), zorder=10)

newlims = [np.min([plt.ylim(), plt.xlim()]), np.max([plt.ylim(), plt.xlim()])]

plt.axis('equal')
plt.axis('square')
plt.xlim(newlims)
plt.ylim(newlims)

plt.plot([1,1],newlims, '--', color=(0.5,0.5,0.5), zorder=0)
plt.plot(newlims,[1,1], '--', color=(0.5,0.5,0.5), zorder=0)
plt.plot(newlims,newlims,'-.', color=(0.5,0.5,0.5), zorder=0)


plt.xlabel('Attention Ratio')
plt.ylabel('Distortion Ratio')

# c, p = sts.pearsonr(atrat.flatten(),drat.flatten())
