SAVE_DIR = '/home/matteo/Documents/github/bertembeddings/data/'
CODE_DIR = '/home/matteo/Documents/github/bertembeddings/'

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
random_model = True

if random_model:
    model = BertModel(BertConfig(output_hidden_states=True, output_attentions=True))
else:
    # model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)
    config = AutoConfig.from_pretrained('bert-base-cased', output_hidden_states=True,
                                    output_attentions=True,
                                    cache_dir='pretrained_models')
    model = AutoModel.from_config(config)
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', cache_dir='pretrained_models')

dfile = SAVE_DIR+'train_bracketed.txt'

# with open(SAVE_DIR+'permuted_data.pkl', 'rb') as dfile:
#     dist = pkl.load(dfile)

#%%
max_num = 300
these_n = [1,2,3,4,5,6,7]
do_shuffling = True

frob = []
nuc = []
inf = []
csim = []
avgdist = []
whichline = []
whichcond = []
whichswap = []
norms = []

num_cond = np.zeros(len(these_n))
t0 = time()
pbar = tqdm.tqdm(total=max_num*len(these_n))
for line_idx in np.random.permutation(range(5000)):
    
    line = linecache.getline(dfile, line_idx+1)
    sentence = BracketedSentence(line)
    if sentence.ntok<10:
        continue
    # orig = d[0]
    orig = sentence.words
    ntok = sentence.ntok
    
    valid = np.floor(ntok/np.array(these_n))>2
    if not np.any(valid):
        continue
    
    orig_idx = np.arange(ntok)
    
    for c in np.random.choice(np.argwhere(valid).squeeze(), 1):
        if (num_cond[c] >= max_num):
            continue
        num_cond[c] += 1
        
        swap_idx = np.arange(ntok)
        if do_shuffling:
            n_pad = int(these_n[c]-np.mod(ntok,these_n[c]))
            # padded = np.insert(swap_idx.astype(float), 
            #                    ntok-n_pad-1, 
            #                    np.ones(n_pad)*np.nan)
            padded = np.append(swap_idx.astype(float), np.ones(n_pad)*np.nan)
            swap_idx = np.random.permutation(padded.reshape((-1,these_n[c]))).flatten()
            swap_idx = swap_idx[~np.isnan(swap_idx)].astype(int)
        swapped = [orig[i] for i in swap_idx]
        
        # real
        orig_vecs = extract_tensor(orig, indices=orig_idx)
        swap_vecs = extract_tensor(swapped, indices=swap_idx)
        
        orig_centred = orig_vecs-orig_vecs.mean(axis=2, keepdims=True)
        swap_centred = swap_vecs-swap_vecs.mean(axis=2, keepdims=True)
        normalizer = (la.norm(orig_centred,2,1,keepdims=True)*la.norm(swap_centred,2,1,keepdims=True))
        csim.append(np.sum((orig_centred*swap_centred)/normalizer,1))
        
        frob.append(la.norm(orig_vecs-swap_vecs,'fro',axis=(1,2))/np.sqrt(ntok))
        nuc.append(la.norm(orig_vecs-swap_vecs,'nuc',axis=(1,2))/np.sqrt(ntok))
        inf.append(la.norm(orig_vecs-swap_vecs, np.inf,axis=(1,2))/np.sqrt(ntok))
        avgdist.append(la.norm(orig_vecs-swap_vecs, 2, axis=1).mean(1))
        
        norms.append(la.norm(np.append(orig_vecs, swap_vecs, -1), 2, axis=1).mean(1))
        
        whichline.append(line_idx)
        whichcond.append(these_n[c])
        whichswap.append(np.repeat(len(whichline), ntok))
        
        pbar.update(1)
    
    if np.all(num_cond >= max_num):
        break
    # print('Done with line %d in %.3f seconds'%(i,time()-t0))

fold = 'ngram_swaps/'
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
np.save(open(SAVE_DIR+fold+'_average_norms.npy','wb'), np.stack(norms))
np.save(open(SAVE_DIR+fold+'_dist_avg.npy','wb'), np.stack(avgdist))

# print('Done!'



