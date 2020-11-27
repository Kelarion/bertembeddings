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
jon_folder = 'C:/Users/mmall/Documents/github/bertembeddings/data/jonathans/ngram/'
random_model = False
# random_model = True

if random_model:
    model = 'bert-base-cased_untrained/'
else:
    model = 'bert-base-cased/'

lines = os.listdir(jon_folder+model)

#%%
# max_num = 200
grams = ['one_gram','bigram','trigram','quadragram','pentagram','sixgram','sevengram']

frob = []
nuc = []
inf = []
csim = []
avgdist = []
whichline = []
whichcond = []
whichswap = []
norms = []
# mean = np.zeros((13, 768))

# compute mean and variance for z-scoring
print('Computing mean and variance ...')
all_vecs = []
for line in lines:
    
    orig_vecs = pkl.load(open(jon_folder+model+line+'/normal_vectors.pkl','rb'))
    which_n = np.random.choice(grams)
    swap_vecs = pkl.load(open(jon_folder+model+line+'/%s_vectors.pkl'%which_n,'rb'))

    catted = np.append(orig_vecs, swap_vecs, -1)
    # means.append(catted.mean(-1))
    # var.append(((catted-catted.mean(-1,keepdims=True))**2).mean(-1))
    all_vecs.append(catted)
m = np.concatenate(all_vecs,-1).mean(-1,keepdims=True)
s = np.concatenate(all_vecs,-1).std(-1,keepdims=True)

# layer_std = np.concatenate(all_vecs,-1).std(axis=(1,2))

t0 = time()
# pbar = tqdm.tqdm(total=max_num*len(these_n))
for line_idx, line in enumerate(tqdm.tqdm(lines)):
    
    orig_vecs = pkl.load(open(jon_folder+model+line+'/normal_vectors.pkl','rb'))
    ntok = orig_vecs.shape[-1]
    
    for g, gram in enumerate(grams):
        
        swap_vecs = pkl.load(open(jon_folder+model+line+'/%s_vectors.pkl'%gram,'rb'))
        
        orig_centred = orig_vecs - m
        swap_centred = swap_vecs - m
        normalizer = (la.norm(orig_centred,2,1,keepdims=True)*la.norm(swap_centred,2,1,keepdims=True))
        csim.append(np.sum((orig_centred*swap_centred)/normalizer,1))
        
        orig_vecs_zscore = (orig_vecs-m)/s
        swap_vecs_zscore = (swap_vecs-m)/s
    
        diff = orig_vecs_zscore-swap_vecs_zscore
        
        frob.append(la.norm(diff, 'fro', axis=(1,2))/np.sqrt(np.prod(diff.shape[1:])))
        nuc.append(la.norm(diff, 'nuc', axis=(1,2))/np.sqrt(np.prod(diff.shape[1:])))
        inf.append(la.norm(diff, np.inf, axis=(1,2))/np.sqrt(np.prod(diff.shape[1:])))
        avgdist.append(la.norm(diff, 2, axis=1).mean(1))
        
        norms.append(la.norm(np.append(orig_centred, swap_centred, -1), 2, axis=1))
        # norms.append(np.append(orig_vecs, swap_vecs, -1).sum(-1))
        # norms.append(la.norm(-2*m/s, 'fro', axis=(1,2))/np.sqrt(ntok))
        
        whichline.append(line_idx)
        whichcond.append(g)
        whichswap.append(np.repeat(len(whichline), ntok))
            
    # if np.all(num_cond >= max_num):
    #     break
    # print('Done with line %d in %.3f seconds'%(i,time()-t0))

fold = 'ngram_swaps/jonathans/'
if random_model:
    fold += 'random_model/'

if not os.path.isdir(SAVE_DIR+fold):
    os.makedirs(SAVE_DIR+fold)

# pref = '%s_%dorder'%(phrase_type, order)

# norms = np.concatenate(norms,-1)

np.save(open(SAVE_DIR+fold+'_cosines.npy','wb'),np.concatenate(csim,axis=1))
np.save(open(SAVE_DIR+fold+'_frob.npy','wb'),np.stack(frob))
np.save(open(SAVE_DIR+fold+'_nuc.npy','wb'),np.stack(nuc))
np.save(open(SAVE_DIR+fold+'_inf.npy','wb'),np.stack(inf))
np.save(open(SAVE_DIR+fold+'_line_id.npy','wb'),whichline)
np.save(open(SAVE_DIR+fold+'_condition.npy','wb'), whichcond)
np.save(open(SAVE_DIR+fold+'_swap_id.npy','wb'), np.concatenate(whichswap))
np.save(open(SAVE_DIR+fold+'_average_norms.npy','wb'), np.concatenate(norms,-1))
# np.save(open(SAVE_DIR+fold+'_average_norms.npy','wb'), norms)
np.save(open(SAVE_DIR+fold+'_dist_avg.npy','wb'), np.stack(avgdist))

# print('Done!'



