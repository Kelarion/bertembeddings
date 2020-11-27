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
            concatenated_vectors = concatenated_vectors[:, indices].reshape(-1, len(indices))
        layer_vectors.append(concatenated_vectors)
        if get_attn and (layer>0):
            attn_matrices = np.stack(list_of_attn).transpose(1,0,2)
            layer_attns.append(attn_matrices)
        
    if get_attn:
        return np.stack(layer_vectors), np.stack(layer_attns)
    else:
        return np.stack(layer_vectors)

#%%
jon_folder = 'C:/Users/mmall/Documents/github/bertembeddings/data/jonathans/adjacent/'
random_model = False
# random_model = True

if random_model:
    # config = AutoConfig.from_pretrained(pretrained_weights, output_hidden_states=True,
    #                                 output_attentions=args.attention,
    #                                 cache_dir='pretrained_models')
    # model = AutoModel.from_config(config)
    model = BertModel(BertConfig(output_hidden_states=True, output_attentions=True))
else:
    model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


lines = pkl.load(open(jon_folder+'phrase_boundary_tree_dist.pkl','rb'))

#%%
max_num = 300
these_bounds = [0,1,2,3,4]

frob = []
nuc = []
inf = []
csim = []
avgdist = []
whichline = []
whichcond = []
whichswap = []
norms = []
# mean = np.zeros((13,768))

print('Computing mean and variance ...')
all_vecs = []
for line_idx in np.random.choice(range(5000), 500):
    
    line = lines[line_idx]
    if len(line[0])<10:
        continue
    
    orig = line[0]
    ntok = len(orig)
    
    orig_idx = np.arange(ntok)
    
    # swap_idx = np.random.permutation(orig_idx)
    swap_idx = np.arange(ntok)
    i = np.random.choice(ntok-1)
    swap_idx[i] = i+1
    swap_idx[i+1] = i
    swapped = [orig[i] for i in swap_idx]
    
    orig_vecs = extract_tensor(orig, indices=orig_idx)
    swap_vecs = extract_tensor(swapped, indices=swap_idx)
    
    catted = np.append(orig_vecs, swap_vecs, -1)
    # means.append(catted.mean(-1))
    # var.append(((catted-catted.mean(-1,keepdims=True))**2).mean(-1))
    all_vecs.append(catted)
    
m = np.concatenate(all_vecs,-1).mean(-1,keepdims=True)
s = np.concatenate(all_vecs,-1).std(-1,keepdims=True)

num_cond = np.zeros(len(these_bounds))
t0 = time()
pbar = tqdm.tqdm(total=max_num*len(these_bounds))
for line_idx in np.random.permutation(range(len(lines))):
    
    line = lines[line_idx]
    if len(line[0])<10:
        continue
    # orig = d[0]
    orig = line[0]
    ntok = len(orig)

    orig_idx = np.array(range(ntok))
    
    c = line[-1]-2
    i = line[2][0]
    if (c not in these_bounds) or (num_cond[c] >= max_num):
        continue
    num_cond[c] += 1
    
    swap_idx = np.array(range(ntok))
    swap_idx[i+1] = i
    swap_idx[i] = i+1
    
    swapped = line[1]
    
    assert([orig[i] for i in swap_idx] == swapped)
        
    # real
    orig_vecs = extract_tensor(orig, indices=orig_idx)
    swap_vecs = extract_tensor(swapped, indices=swap_idx)
    
    orig_vecs_zscore = (orig_vecs-m)/s
    swap_vecs_zscore = (swap_vecs-m)/s
    
    diff = orig_vecs_zscore-swap_vecs_zscore
    
    orig_centred = orig_vecs-m  #orig_vecs.mean(axis=2, keepdims=True)
    swap_centred = swap_vecs-m  #swap_vecs.mean(axis=2, keepdims=True)
    normalizer = (la.norm(orig_centred,2,1,keepdims=True)*la.norm(swap_centred,2,1,keepdims=True))
    csim.append(np.sum((orig_centred*swap_centred)/normalizer,1))
    
    frob.append(la.norm(diff,'fro',axis=(1,2))/np.sqrt(np.prod(diff.shape[1:])))
    nuc.append(la.norm(diff,'nuc',axis=(1,2))/np.sqrt(np.prod(diff.shape[1:])))
    inf.append(la.norm(diff, np.inf,axis=(1,2))/np.sqrt(np.prod(diff.shape[1:])))
    avgdist.append(la.norm(diff, 2, axis=1).mean(1))
    
    norms.append(la.norm(np.append(orig_centred, swap_centred, -1), 2, -2))
    
    # mean += np.append(orig_centred, swap_centred, -1).sum(-1)
    
    whichline.append(line_idx)
    whichcond.append(c)
    whichswap.append(np.repeat(len(whichline), ntok))
    
    pbar.update(1)
    
    if np.all(num_cond >= max_num):
        break
    # print('Done with line %d in %.3f seconds'%(i,time()-t0))

fold = 'bracket_crossings/full_features/'
if random_model:
    fold += 'random_model/'

if not os.path.isdir(SAVE_DIR+fold):
    os.makedirs(SAVE_DIR+fold)

# pref = '%s_%dorder'%(phrase_type, order)

np.save(open(SAVE_DIR+fold+'_cosines.npy','wb'),np.concatenate(csim,axis=1))
np.save(open(SAVE_DIR+fold+'_frob.npy','wb'),np.stack(frob))
np.save(open(SAVE_DIR+fold+'_nuc.npy','wb'),np.stack(nuc))
np.save(open(SAVE_DIR+fold+'_inf.npy','wb'),np.stack(inf))
np.save(open(SAVE_DIR+fold+'_line_id.npy','wb'),whichline)
np.save(open(SAVE_DIR+fold+'_num_crossings.npy','wb'), whichcond)
np.save(open(SAVE_DIR+fold+'_swap_id.npy','wb'), np.concatenate(whichswap))
np.save(open(SAVE_DIR+fold+'_average_norms.npy','wb'), np.concatenate(norms, -1))
# np.save(open(SAVE_DIR+fold+'_average_norms.npy','wb'), np.concatenate(norms, -1))
np.save(open(SAVE_DIR+fold+'_dist_avg.npy','wb'), np.stack(avgdist))

print('Done!')