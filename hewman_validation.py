SAVE_DIR = 'C:/Users/mmall/Documents/github/bertembeddings/data/'
CODE_DIR = 'C:/Users/mmall/Documents/github/bertembeddings/'

import sys, os
sys.path.append(CODE_DIR)
from brackets2trees import BracketedSentence
from spline_regression import 
from hyperbolic_utils import EuclideanEncoder

import torch
import torch.nn as nn
import torch.optim as optim

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

#%%
N = 100
num_init = 1
num_lines = 500

encoder = nn.Linear(768, N, bias=False)
probe = EuclideanEncoder(encoder)
# criterion = nn.MSELoss(reduction='mean')
# criterion = nn.L1Loss(reduction='mean')
criterion = nn.PoissonNLLLoss(reduction='mean', log_input=False)

# rank_corr = np.zeros((5,13,num_init))
rank_corr = []
# test_loss = np.zeros((5,13,num_init))
for init in range(num_init):
    # swap_idx_all = [[] for _ in range(500)]
    
    for layer in range(13):
        
        expinf = 'layer%d_rank%d_init%d_%s_linear'%(layer, N, init, criterion.__class__.__name__)
        
        num_pos = np.zeros(5)
        num_syn = np.zeros(5)
        
        with open(SAVE_DIR+fold+expinf+'_params.pt', 'rb') as f:
            probe.load_state_dict(torch.load(f))
            
        # dbs = [[] for _ in range(5)]
        # dts = []
        scorr = [[] for _ in range(5)]
        # logli = [[] for _ in range(5)]
        for line_idx in tqdm.tqdm(np.random.choice(5000,num_lines), desc='Layer %d, model %d'%(layer, init)):
            
            line = linecache.getline(dfile, line_idx+1)
            sentence = BracketedSentence(line, dep_tree=dep_tree)
            
            if sentence.ntok<10:
                continue
            if sentence.ntok>110:
                continue
            
            toks = np.arange(sentence.ntok)
            ntok = sentence.ntok
            w1, w2 = np.nonzero(np.triu(np.ones((ntok,ntok)),k=1))
            
            # tree distance
            dT = np.array([sentence.tree_dist(w1[i],w2[i],term=(not dep_tree)) for i in range(len(w1))])
            
            # num_pos[0] += 1
            # num_syn[0] += 1
            
            # get the features
            orig = sentence.words
            ntok = sentence.ntok
            
            orig_idx = np.array(range(ntok))
            orig_vecs = extract_tensor(orig, indices=orig_idx)
            
            # probe distance
            W1 = probe(torch.tensor(orig_vecs[layer,:,w1])) # map onto euclidean or hyperbolic space
            W2 = probe(torch.tensor(orig_vecs[layer,:,w2]))
            
            dB = probe.dist(W1, W2).detach().numpy()
            
            scorr[0].append(sts.spearmanr(dT, dB)[0])
            # logli[0].append(-nn.PoissonNLLLoss(log_input=False)(dB, dT))
            
            # pos_perf[0,layer,init] += (glm_pos(orig_vecs[...,valid_pos].T).argmax(1) == y_pos).sum().float()
            # syn_perf[0,layer,init] += (glm_syn(orig_vecs[...,valid_syn].T).argmax(1) == y_syn).sum().float()
            
            for t, tree_dist in enumerate(range(2,6)):
                # find all swaps of the desired tree_distance
                
                valid = dT[np.abs(np.diff(w1,prepend=True))==1] == tree_dist
                
                if not np.any(valid):
                    continue

                swp = np.array(range(ntok))
                i = np.random.choice(np.nonzero(valid)[0],1)
                swp[i] = i+1
                swp[i+1] = i
                assert(sentence.tree_dist(i,i+1,term=(not dep_tree))==tree_dist)
                swap_idx = swp
                
                swap_vecs = extract_tensor([orig[s] for s in swap_idx], indices=swap_idx)
                
                W1 = probe(torch.tensor(swap_vecs[layer,:,w1])) # map onto euclidean or hyperbolic space
                W2 = probe(torch.tensor(swap_vecs[layer,:,w2]))
                
                dB = probe.dist(W1, W2).detach().numpy()
                
                scorr[t+1].append(sts.spearmanr(dT, dB)[0])
                # logli[t+1].append(-nn.PoissonNLLLoss(log_input=False)(dB, dT).item())
                
        rank_corr.append(scorr[0])
        # test_loss[:,layer,init] = [np.mean(s) for s in logli]

#%%
plot_this = rank_corr

cmap = cm.get_cmap('viridis')

col = cmap(np.arange(5)/5)

for t in range(5):
    plt.plot(plot_this[t,:,:].mean(1),marker='.', color=col[t])
    plt.fill_between(range(13),
                     plot_this[t,:,:].mean(1)-plot_this[t,:,:].std(1),
                     plot_this[t,:,:].mean(1)+plot_this[t,:,:].std(1),
                     color=col[t],
                     alpha=0.5)

plt.legend(['Unswapped',2,3,4,5],title='Tree distance')
plt.xlabel('Layer')
plt.ylabel('Rank correlation')

