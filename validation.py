SAVE_DIR = 'C:/Users/mmall/Documents/github/bertembeddings/data/'
CODE_DIR = 'C:/Users/mmall/Documents/github/bertembeddings/'

import sys, os
sys.path.append(CODE_DIR)
from brackets2trees import BracketedSentence

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D

import tqdm

from transformers import BertTokenizer, BertModel, BertConfig, AutoConfig, AutoModel, AutoTokenizer
import pickle as pkl
import numpy as np
import scipy.linalg as la
import linecache
from time import time
import matplotlib.pyplot as plt
from matplotlib import cm
from cycler import cycler

fold = 'linear_probes/'

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

if random_model:
    model = BertModel(BertConfig(output_hidden_states=True, output_attentions=True, cache_dir='pretrained_models'))
else:
    model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    
dfile = SAVE_DIR+'train_bracketed.txt'

pos_tags = list(np.load(SAVE_DIR+'unique_pos_tags.npy'))
phrase_tags = list(np.load(SAVE_DIR+'unique_phrase_tags.npy'))

# for mysterious purposes
sd = np.load(SAVE_DIR+'bracket_crossings/full_features/global_sd.npy')
frob = np.load(SAVE_DIR+'bracket_crossings/full_features/_frob.npy')

#%%
# these_inits = list(range(10))
these_inits = [0]
num_lines = 600
num_tree = 5
# full_sentence = False
full_sentence = True

glm_pos = nn.Linear(768, len(pos_tags), bias=True)
glm_syn = nn.Linear(768, len(phrase_tags), bias=True)


pos_perf = np.zeros((5,13,len(these_inits)))
syn_perf = np.zeros((5,13,len(these_inits)))
# distortion = [[] for _ in range(13)]
distortion = []
pos_logli = [[] for _ in range(13)] # loglihood on training context
pos_swp_logli = [[] for _ in range(13)] # cross-context 
syn_logli = [[] for _ in range(13)]
syn_swp_logli = [[] for _ in range(13)]
# dtree = [[] for _ in range(13)]
dtree = []
# for init in these_inits:
#     # swap_idx_all = [[] for _ in range(500)]
    
#     for layer in range(13):
        
#         num_pos = np.zeros(5)
#         num_syn = np.zeros(5)
        
#         with open(SAVE_DIR+fold+'pos_layer%d_init%d_params.pt'%(layer,init), 'rb') as f:
#             glm_pos.load_state_dict(torch.load(f))
#         with open(SAVE_DIR+fold+'syn_layer%d_init%d_params.pt'%(layer,init), 'rb') as f:
#             glm_syn.load_state_dict(torch.load(f))
            
for line_idx in tqdm.tqdm(np.random.choice(5000,num_lines,replace=False)):
    
    line = linecache.getline(dfile, line_idx+1)
    sentence = BracketedSentence(line)
    
    if sentence.ntok<10:
        continue
    
    # get the POS and Grandparent tag labels
    labels_pos = np.array([pos_tags.index(w) if w in pos_tags else np.nan for w in sentence.pos_tags])
    valid_pos = ~np.isnan(labels_pos)
    y_pos = torch.tensor(labels_pos[valid_pos]).long()
    
    anc_tags = [sentence.ancestor_tags(i, 2) for i in range(sentence.ntok)]
    labels_syn = np.array([phrase_tags.index(w) if w in phrase_tags else np.nan for w in anc_tags])
    valid_syn = ~np.isnan(labels_syn)
    y_syn = torch.tensor(labels_syn[valid_syn]).long()
    
    all_valid = valid_pos*valid_syn
    pos_tags_int = torch.tensor(labels_pos).long()
    syn_tags_int = torch.tensor(labels_syn).long()
    
    # num_pos[0] += sum(valid_pos)
    # num_syn[0] += sum(valid_syn)
    
    # get the features
    orig = sentence.words
    ntok = sentence.ntok
    
    orig_idx = np.array(range(ntok))
    orig_vecs = torch.tensor(extract_tensor(orig, indices=orig_idx))
            
    # pos_logli[layer].append(orig_pos_logli[i])
    # pos_logli[layer].append(orig_pos_logli[i+1])
    # syn_logli[layer].append(orig_syn_logli[i])
    # syn_logli[layer].append(orig_syn_logli[i+1])
    
    for t, tree_dist in enumerate(range(2,2+num_tree)):
        # find all swaps of the desired tree_distance
        valid = np.array([sentence.tree_dist(i,i-1) for i in range(1,sentence.ntok)]) == tree_dist
        
        # we need to make sure that both swapped words have valid POS and grandparent tags
        if not np.any(valid*all_valid[1:]*all_valid[:-1]):
            continue
    
        # num_pos[t+1] += sum(valid_pos)
        # num_syn[t+1] += sum(valid_syn)
        
        swp = np.array(range(ntok))
        i = int(np.random.choice(np.nonzero(valid*all_valid[1:]*all_valid[:-1])[0],1))
        swp[i] = i+1
        swp[i+1] = i
        # assert(sentence.tree_dist(i,i+1)==tree_dist)
        swap_idx = swp
        
        swap_vecs = torch.tensor(extract_tensor([orig[s] for s in swap_idx], indices=swap_idx))
        
        if full_sentence:
            diff = (orig_vecs-swap_vecs)/sd
            dist = la.norm(diff, 'fro', axis=(1,2))/np.sqrt(np.prod(diff.shape[1:]))
        else:
            diff = (orig_vecs[:,:,i:i+2]-swap_vecs[:,:,i:i+2])/sd
            dist = la.norm(diff, 2, 1)/np.sqrt(diff.shape[1])
        distortion.append(dist)
        
        dtree.append(tree_dist)
        
        for init in these_inits:
            for layer in range(13):
                
                # num_pos = np.zeros(5)
                # num_syn = np.zeros(5)
                
                with open(SAVE_DIR+fold+'pos_layer%d_init%d_params.pt'%(layer,init), 'rb') as f:
                    glm_pos.load_state_dict(torch.load(f))
                with open(SAVE_DIR+fold+'syn_layer%d_init%d_params.pt'%(layer,init), 'rb') as f:
                    glm_syn.load_state_dict(torch.load(f))
        
                # pos_perf[0,layer,init] += (glm_pos(orig_vecs[layer,:,valid_pos].T).argmax(1) == y_pos).sum().float()
                # syn_perf[0,layer,init] += (glm_syn(orig_vecs[layer,:,valid_syn].T).argmax(1) == y_syn).sum().float()
                
                if full_sentence:
                    idx = all_valid
                else:
                    idx = [i,i+1]
                pos_model = D.categorical.Categorical(logits=glm_pos(orig_vecs[layer,:,idx].T))
                syn_model = D.categorical.Categorical(logits=glm_syn(orig_vecs[layer,:,idx].T))
                
                orig_pos_logli = pos_model.log_prob(pos_tags_int[idx])
                orig_syn_logli = syn_model.log_prob(syn_tags_int[idx])
                # pos_perf[t+1,layer,init] += (glm_pos(swap_vecs[...,valid_pos].T).argmax(1) == y_pos).sum().float()
                # syn_perf[t+1,layer,init] += (glm_syn(swap_vecs[...,valid_syn].T).argmax(1) == y_syn).sum().float()
                
                pos_logli[layer].append(orig_pos_logli.detach().numpy())
                # pos_swp_logli[layer].append(pos_delta[i+1])
                syn_logli[layer].append(orig_syn_logli.detach().numpy())
                
                pos_swp_model = D.categorical.Categorical(logits=glm_pos(swap_vecs[layer,:,idx].T))
                syn_swp_model = D.categorical.Categorical(logits=glm_syn(swap_vecs[layer,:,idx].T))
                
                pos_delta = pos_swp_model.log_prob(pos_tags_int[idx])#-orig_pos_logli
                syn_delta = syn_swp_model.log_prob(syn_tags_int[idx])#-orig_syn_logli
                
                pos_swp_logli[layer].append(pos_delta.detach().numpy())
                # pos_swp_logli[layer].append(pos_delta[i+1])
                syn_swp_logli[layer].append(syn_delta.detach().numpy())
                # syn_swp_logli[layer].append(syn_delta[i+1])
                
                # dtree[layer].append(sentence.tree_dist(i,i+1))
        
        
        # pos_perf[:,layer,init] /= num_pos
        # syn_perf[:,layer,init] /= num_syn

# np.save(SAVE_DIR+'probe_results/pos_perf.npy', pos_perf)
# np.save(SAVE_DIR+'probe_results/syn_perf.npy', syn_perf)

if full_sentence:
    d = np.stack(distortion)
    # num_word = []
else:
    d = np.stack(distortion).transpose((0,2,1)).reshape((-1,13))
num_word = [l.shape[0] for l in pos_logli[0]]
dtree = np.array(dtree)

#%% Depth series
logq = pos_swp_logli
logp = pos_logli

# logq = syn_swp_logli
# logp = syn_logli

cmap = cm.get_cmap('viridis')

col = cmap(np.arange(num_tree)/num_tree)

t_ = np.repeat(dtree,num_word)    
    
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

#%% Volcano plot
# logq = pos_swp_logli
# logp = pos_logli

logq = syn_swp_logli
logp = syn_logli

cmap = cm.get_cmap('viridis')

col = cmap(np.arange(13)/13)

for i in range(13):
    logqp = np.concatenate(logq[i])-np.concatenate(logp[i])
    plt.scatter(logqp, np.repeat(d[:,i],num_word), alpha=0.5, s=1, c=[col[i]])

plt.colorbar(ticks=range(13),drawedges=True,values=range(13), label='Layer')
plt.xlabel(r'Change in log-likelihood ($\log{\frac{q}{p}}$)')
plt.ylabel('Distortion (word-wise)')

#%% Averaging things
cmap_tree = 'viridis'
# cmap_layer = 'plasma'
cmap_layer = 'binary'

# show_baseline = False
show_baseline = True

logq = pos_swp_logli
logp = pos_logli

# logq = syn_swp_logli
# logp = syn_logli

cmap = cm.get_cmap(cmap_layer)

col = cmap(np.arange(13)/13)

ax = plt.axes()

mean_perf = []
mean_d = []
scats = []
for i in range(13):
    
    t_ = np.repeat(dtree,num_word)    
    
    if full_sentence:
        means_dist = np.array([d[dtree==t,i].mean() for t in np.unique(dtree)])
        err_dist = np.array([d[dtree==t,i].std()/np.sqrt(np.sum(dtree==t)) for t in np.unique(dtree)])
    else:
        means_dist = np.array([d[t_==t,i].mean() for t in np.unique(dtree)])
        err_dist = np.array([d[t_==t,i].std()/np.sqrt(np.sum(dtree==t)) for t in np.unique(dtree)])
        
    if show_baseline:
        mean0 = np.concatenate(logp[i]).mean()
        err0 = np.concatenate(logp[i]).std()/np.sqrt(len(logp[i]))
        means_logli = mean0
        means_dist = np.mean(means_dist)
        # means_logli = np.array([mean0]+[np.concatenate(logq[i])[t_==t].mean() for t in np.unique(dtree)])
        # err_logli = np.array([err0]+[np.concatenate(logq[i])[t_==t].std()/np.sqrt(np.sum(dtree==t)) \
        #                              for t in np.unique(dtree)])
        # means_dist = np.append(np.mean(means_dist),means_dist)
        # err_dist = np.append(0,err_dist)
    else:
        logqp = np.concatenate(logp[i])-np.concatenate(logq[i])
        means_logli = np.array([logqp[t_==t].mean() for t in np.unique(dtree)])
        err_logli = np.array([logqp[t_==t].std()/np.sqrt(np.sum(dtree==t)) for t in np.unique(dtree)])
    
    mean_perf.append(means_logli)
    mean_d.append(means_dist)
    
    plt.scatter(means_logli, means_dist, marker='o', s=30, c=[col[i]], edgecolors='k')
    # plt.scatter(means_logli, means_dist, s=30, c=[col[i]])
    # sct = plt.errorbar(means_logli, means_dist, 
    #                      xerr=err_logli,
    #                      yerr=err_dist,
    #                      linestyle='none',
    #                      ecolor=col[i])
    # scats.append(sct)

plt.colorbar(cm.ScalarMappable(cmap=cmap), 
             ticks=np.arange(13),
             drawedges=True,
             values=np.arange(13), 
             label='Layer')
plt.xlabel(r'Average decrease in log-likelihood')
plt.ylabel('Average distortion')

x = np.stack(mean_perf)
y = np.stack(mean_d)

cols2 = cm.get_cmap(cmap_tree)(np.arange(num_tree)/num_tree)
# if show_baseline:
#     cols2 = 
ax.set_prop_cycle(cycler(color=cols2))
lines = plt.plot(x,y, zorder=0, linewidth=2)

plt.legend(lines,np.unique(dtree),title='Tree distance')




