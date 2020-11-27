SAVE_DIR = 'C:/Users/mmall/Documents/github/bertembeddings/data/'
CODE_DIR = 'C:/Users/mmall/Documents/github/bertembeddings/'

import sys, os
sys.path.append(CODE_DIR)
from brackets2trees import BracketedSentence
from hyperbolic_utils import EuclideanEncoder

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D

import tqdm

from transformers import BertTokenizer, BertModel, BertConfig, AutoConfig, AutoModel, AutoTokenizer
import pickle as pkl
import numpy as np
import scipy.linalg as la
import scipy.stats as sts
import linecache
from time import time
import matplotlib.pyplot as plt
from matplotlib import cm
from cycler import cycler

#%%
def rindex(alist, value):
    return len(alist) - alist[-1::-1].index(value) - 1

def extract_tensor(text_array, indices=None, num_layers=13, get_attn=False):
    layers = range(num_layers)

    word_tokens = []
    split_word_idx = []
    for split_id, split_word in enumerate(text_array):
        tokens = tokenizer.tokenize(split_word)
        word_tokens.extend(tokens)
        split_word_idx.extend([split_id + 1] * len(tokens))

    input_ids = torch.Tensor([tokenizer.encode(word_tokens, add_special_tokens=True)]).long()

    # Getting torch output
    with torch.no_grad():
        rtn = model(input_ids)
        bert_output = rtn[2]
        attn_weight = rtn[3]
        # attn_weight = model(input_ids)[3]
        
    # Index of sorted line
    layer_vectors = []
    layer_attns = []
    for layer in layers:
        list_of_vectors = []
        list_of_attn = []
        for word_idx in range(len(text_array)):
            this_word_idx = word_idx + 1

            # the average vector for the subword will be used
            vector_idcs = np.argwhere(np.array(split_word_idx) == this_word_idx).reshape(-1) + 1
            token_vector = bert_output[layer][0][vector_idcs].mean(0).cpu().reshape(-1, 1).numpy()
            
            if get_attn and (layer>0):
                attn_vectors = attn_weight[layer-1][0][:,vector_idcs,:].mean(1).cpu().numpy()
                
                # need to take two average for attention
                attn_mean = []
                for word_idx in range(len(text_array)):
                    this_word_idx = word_idx + 1
                    vector_idcs = np.argwhere(np.array(split_word_idx) == this_word_idx).reshape(-1) + 1
                    attn_mean.append(attn_vectors[:,vector_idcs].mean(1))
                list_of_attn.append(np.array(attn_mean).T)
            list_of_vectors.append(token_vector)

        concatenated_vectors = np.concatenate(list_of_vectors, 1)
        if indices is not None:
            # print(concatenated_vectors.shape)
            concatenated_vectors = concatenated_vectors[:, indices].reshape(-1, len(indices))
            # print(indices)
            # print(concatenated_vectors.shape)
        layer_vectors.append(concatenated_vectors)
        if get_attn and (layer>0):
            attn_matrices = np.stack(list_of_attn).transpose(1,0,2)
            layer_attns.append(attn_matrices)
        
    if get_attn:
        return np.stack(layer_vectors), np.stack(layer_attns)
    else:
        return np.stack(layer_vectors)

#%%
# # random_model = True
random_model = False

# dep_tree = True
dep_tree = False

if random_model:
    model = BertModel(BertConfig(output_hidden_states=True, output_attentions=True, cache_dir='pretrained_models'))
else:
    model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True, output_attentions=True)
    # config = AutoConfig.from_pretrained('bert-base-cased', output_hidden_states=True,
    #                                 output_attentions=True,
    #                                 cache_dir='pretrained_models')
    # model = AutoModel.from_config(config)
# tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', cache_dir='pretrained_models')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

if dep_tree:
    dfile = SAVE_DIR+'dependency_train_bracketed.txt'
    idx = np.load(SAVE_DIR+'const_in_dep.npy').astype(int)
    fold = '/dep/EuclideanEncoder/'
else:
    dfile = SAVE_DIR+'train_bracketed.txt'
    idx = np.arange(5000)
    fold = '/const/EuclideanEncoder/'

const_dfile = SAVE_DIR+'train_bracketed.txt'

sd = np.load(SAVE_DIR+'bracket_crossings/full_features/global_sd.npy')

#%%
N = 100
# num_init = 10
num_init = 4
num_line = 500
num_tree = 5

full_sentence = True
# full_sentence = False

encoder = nn.Linear(768, N, bias=False)
probe = EuclideanEncoder(encoder)
# criterion = nn.MSELoss(reduction='mean')
# criterion = nn.L1Loss(reduction='mean')
criterion = nn.PoissonNLLLoss(reduction='mean', log_input=False)

distortion = []
logli = [[] for _ in range(13)] # loglihood on training context
swp_logli = [[] for _ in range(13)] # cross-context
dtree = []
for line_idx in tqdm.tqdm(np.random.choice(5000,num_line)):
    
    line = linecache.getline(dfile, line_idx+1)
    sentence = BracketedSentence(line, dep_tree=dep_tree)
    
    if sentence.ntok<10:
        continue
    if sentence.ntok>110:
        continue

    toks = np.arange(sentence.ntok)
    ntok = sentence.ntok
    
    if full_sentence:
        w1, w2 = np.nonzero(np.triu(np.ones((ntok,ntok)),k=1))
        dT = np.array([sentence.tree_dist(w1[i],w2[i],term=(not dep_tree)) for i in range(len(w1))])
    
    # num_pos[0] += 1
    # num_syn[0] += 1
    
    # get the features
    orig = sentence.words
    ntok = sentence.ntok
    
    orig_idx = np.array(range(ntok))
    orig_vecs = extract_tensor(orig, indices=orig_idx)
    
    # scorr[0].append(sts.spearmanr(dT, dB)[0])
    
    # pos_perf[0,layer,init] += (glm_pos(orig_vecs[...,valid_pos].T).argmax(1) == y_pos).sum().float()
    # syn_perf[0,layer,init] += (glm_syn(orig_vecs[...,valid_syn].T).argmax(1) == y_syn).sum().float()
    
    for t, tree_dist in enumerate(np.arange(2,2+num_tree)-1*dep_tree):
        # find all swaps of the desired tree_distance
        
        valid = np.array([sentence.tree_dist(i,i-1,term=(not dep_tree)) for i in range(1,sentence.ntok)]) == tree_dist
        
        if not np.any(valid):
            # print('Skipping')
            continue

        swp = np.array(range(ntok))
        i = int(np.random.choice(np.nonzero(valid)[0],1))
        swp[i] = i+1
        swp[i+1] = i
        assert(sentence.tree_dist(i,i+1,term=(not dep_tree))==tree_dist)
        swap_idx = swp
        
        if not full_sentence:
            w1 = np.array([i])
            w2 = np.array([i+1])
            dT = np.array([sentence.tree_dist(w1[i],w2[i],term=(not dep_tree)) for i in range(len(w1))]).squeeze()
        
        swap_vecs = extract_tensor([orig[s] for s in swap_idx], indices=swap_idx)
        
        # diff = (orig_vecs[:,:,i:i+2]-swap_vecs[:,:,i:i+2])/sd
        diff = (orig_vecs-swap_vecs)/sd
        dist = la.norm(diff, 'fro', axis=(1,2))/np.sqrt(np.prod(diff.shape[1:]))
        distortion.append(dist)
        
        dtree.append(tree_dist)
        
        for init in range(num_init):
            for layer in range(13):
                
                expinf = 'layer%d_rank%d_init%d_%s_linear'%(layer, N, init, criterion.__class__.__name__)
                
                # num_pos = np.zeros(5)
                # num_syn = np.zeros(5)
                
                with open(SAVE_DIR+fold+expinf+'_params.pt', 'rb') as f:
                    probe.load_state_dict(torch.load(f))
               
                W1 = probe(torch.tensor(orig_vecs[layer,:,w1])) # map onto euclidean or hyperbolic space
                W2 = probe(torch.tensor(orig_vecs[layer,:,w2]))
                
                orig_model = D.poisson.Poisson(probe.dist(W1, W2))
                # logli[layer].append(orig_model.log_prob(torch.tensor(dT).float()).item())
                logli[layer].append(orig_model.log_prob(torch.tensor(dT).float()).detach().numpy())
                # logli[layer] += orig_model(dT)
                
                W1_swp = probe(torch.tensor(swap_vecs[layer,:,w1])) # map onto euclidean or hyperbolic space
                W2_swp = probe(torch.tensor(swap_vecs[layer,:,w2]))
                
                swp_model = D.poisson.Poisson(probe.dist(W1_swp, W2_swp))
                # swp_logli[layer].append(swp_model.log_prob(torch.tensor(dT).float()).item())
                swp_logli[layer].append(swp_model.log_prob(torch.tensor(dT).float()).detach().numpy())
                
                # scorr[t+1].append(sts.spearmanr(dT, dB)[0])
            
        # rank_corr[:,layer,init] = [np.mean(s) for s in scorr]

d = np.stack(distortion)
dtree = np.stack(dtree)
# d = np.concatenate(distortion,-1)
# dtree = np.concatenate(dtree)
num_word = [len(l) for l in swp_logli[0]]

#%% Layerwise plot

logq = swp_logli
logp = logli

cmap = cm.get_cmap('viridis')

col = cmap(np.arange(num_tree)/num_tree)

# t_ = np.repeat(dtree,num_word)
# t_ = np.repeat(dtree,num_word)
t_ = np.repeat(np.repeat(dtree,num_init),num_word)
    
orig_perf = np.array([np.concatenate(logp[i]).mean() for i in range(13)])
orig_err = np.array([np.concatenate(logp[i]).std()/np.sqrt(len(t_)) for i in range(13)])
plt.plot(orig_perf, marker='.', color='k')
plt.fill_between(range(13),orig_perf-orig_err, orig_perf+orig_err,color='k', alpha=0.5)
for t in range(2,2+num_tree):
    swap_perf = np.array([np.concatenate(logq[i])[t_==t].mean() for i in range(13)])
    swap_err = np.array([np.concatenate(logq[i])[t_==t].std()/np.sqrt(np.sum(t_==t)) for i in range(13)])
    plt.plot(swap_perf,marker='.', color=col[t-2])
    plt.fill_between(range(13),swap_perf-swap_err, swap_perf+swap_err,color=col[t-2], alpha=0.5)

plt.legend(['Unswapped'] + list(range(2,2+num_tree)),title='Tree distance')
plt.xlabel('Layer')
plt.ylabel('Average likelihood')


#%% 'Volcano' plot
# logq = swp_logli
# logp = logli

logq = [[l.mean() for l in swp_logli[i]] for i in range(13)]
logp = [[l.mean() for l in logli[i]] for i in range(13)]
num_word = [len(l) for l in swp_logli[0]]

cmap = cm.get_cmap('viridis')

col = cmap(np.arange(13)/13)

for i in range(13):
    plt.scatter(np.array(logq[i])-np.array(logp[i]), d[:,i],alpha=0.5, s=1, c=[col[i]])

plt.colorbar(ticks=range(13),drawedges=True,values=range(13), label='Layer')
plt.xlabel(r'Change in log-likelihood ($\log{\frac{q}{p}}$)')
plt.ylabel('Distortion (sentence level)')

#%% Doing some averaging
cmap_tree = 'viridis'
cmap_layer = 'binary'

logq = swp_logli
logp = logli

cmap = cm.get_cmap(cmap_layer)
col = cmap(np.arange(13)/13)

ax = plt.axes()
mean_perf = []
mean_d = []
for i in range(13):
    t_ = np.repeat(np.repeat(dtree,num_init),num_word)
    
    logqp = np.concatenate(logp[i])-np.concatenate(logq[i])
    means_logli = np.array([logqp[t_==t].mean() for t in np.unique(dtree)])
    means_dist = np.array([d[dtree==t,i].mean() for t in np.unique(dtree)])
    
    mean_perf.append(means_logli)
    mean_d.append(means_dist)
    
    plt.scatter(means_logli, means_dist, marker='o', s=30, c=[col[i]], edgecolors='k')

plt.colorbar(cm.ScalarMappable(cmap=cmap), 
             ticks=np.arange(13),
             drawedges=True,
             values=np.arange(13), 
             label='Layer')
plt.xlabel(r'Average decrease in log-likelihood')
plt.ylabel('Average distortion')

cols2 = cm.get_cmap(cmap_tree)(np.arange(num_tree)/num_tree)
ax.set_prop_cycle(cycler(color=cols2))
x = np.stack(mean_perf)
y = np.stack(mean_d)
lines = plt.plot(x,y, zorder=0, linewidth=2)
plt.legend(lines,np.unique(dtree),title='Tree distance')

# plt.plot(x,y, zorder=0)
# plt.legend(np.unique(dtree),title='Tree distance')

