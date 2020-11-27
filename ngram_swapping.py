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

def extract_tensor(text_array, indices=None, num_layers=13, get_attn=False, split_words=False):
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
    if split_words:
        vecs = np.concatenate(bert_output)[:,1:-1,:].transpose((0,2,1))
        word_bounds = np.unique(split_word_idx,return_index=True)[1]
        chunks = np.array(np.split(np.arange(len(split_word_idx)),word_bounds))[1:]
        if indices is None:
            unpermute = np.arange(len(split_word_idx))
        else:
            unpermute = np.concatenate(chunks[indices])
        return vecs[:,:,unpermute]
    
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

def ngram_shuffling(idx, n):
    """ 
    idx is a n_token length sequence of indices, n is the size of a chunk 
    ensure that n<len(idx)
    """
    if len(idx)<n:
        raise ValueError
    n_pad = int(n-np.mod(len(idx),n))
    # padded = np.insert(swap_idx.astype(float), 
    #                    ntok-n_pad-1, 
    #                    np.ones(n_pad)*np.nan)
    padded = np.append(idx.astype(float), np.ones(n_pad)*np.nan)
    while 1:
        shuf_idx = np.random.permutation(padded.reshape((-1,n))).flatten()
        shuf_idx = shuf_idx[~np.isnan(shuf_idx)].astype(int)
        if np.any(shuf_idx != idx):
            break
    # shuf_idx = np.random.permutation(padded.reshape((-1,n))).flatten()
    # shuf_idx = shuf_idx[~np.isnan(shuf_idx)].astype(int)
    return shuf_idx

#%%
# random_model = False
random_model = True

if random_model:
    # config = AutoConfig.from_pretrained(pretrained_weights, output_hidden_states=True,
    #                                 output_attentions=args.attention,
    #                                 cache_dir='pretrained_models')
    # model = AutoModel.from_config(config)
    model = BertModel(BertConfig(output_hidden_states=True, output_attentions=True))
else:
    model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

dfile = SAVE_DIR+'train_bracketed.txt'

# with open(SAVE_DIR+'permuted_data.pkl', 'rb') as dfile:
#     dist = pkl.load(dfile)

#%%
max_num = 200
these_n = [1,2,3,4,5,6,7]
use_subwords = False
# use_subwords = True

minimum_size = False # should there be at le1ast 2 n-grams in a sentence?
minimum_size = True

frob = []
nuc = []
inf = []
csim = []
whichline = []
whichcond = []
whichswap = []
num_tok = []
num_swaps = []
dist_swaps = []

# compute mean and variance for z-scoring
print('Computing mean and variance ...\n')
# foo = []
all_vecs = []
for line_idx in tqdm.tqdm(np.random.choice(5000,500,replace=False)):
    
    line = linecache.getline(dfile, line_idx+1)
    sentence = BracketedSentence(line)
    orig = sentence.words
    ntok = sentence.ntok
    if ntok < 10:
        continue
    # orig = d[0]
        
    orig_idx = np.array(range(ntok))
    
    n = np.random.choice(these_n)
    # n=1
    swap_idx = ngram_shuffling(np.arange(ntok), n)
    swapped = [orig[i] for i in swap_idx]
    
    # assert(swapped == line[1+phrase_type][swap_type][2])
    assert([swapped[i] for i in np.argsort(swap_idx)] == orig)
    
    # real
    orig_vecs = extract_tensor(orig, split_words=use_subwords)
    swap_vecs = extract_tensor(swapped, indices=np.argsort(swap_idx), split_words=use_subwords)
    
    all_vecs.append(np.append(orig_vecs, swap_vecs, -1))
m = np.concatenate(all_vecs,-1).mean(-1,keepdims=True)
s = np.concatenate(all_vecs,-1).std(-1,keepdims=True)


num_cond = np.zeros(len(these_n))
t0 = time()
pbar = tqdm.tqdm(total=max_num*len(these_n))
for line_idx in np.random.permutation(range(5000)):
    
    line = linecache.getline(dfile, line_idx+1)
    sentence = BracketedSentence(line)
    orig = sentence.words
    ntok = sentence.ntok
    if ntok < 10:
        continue
    # orig = d[0]
    
    valid = (num_cond < max_num)
    if minimum_size:
        valid *= (np.floor(ntok/np.array(these_n))>=2)
    if not np.any(valid):
        continue
    
    orig_idx = np.arange(ntok)
    orig_vecs = extract_tensor(orig, split_words=use_subwords)
    orig_vecs_zscore = (orig_vecs-m)/s
    
    c = np.random.choice(np.where(valid)[0])

    num_cond[c] += 1
    
    swap_idx = ngram_shuffling(np.arange(ntok), these_n[c])
    swapped = [orig[i] for i in swap_idx]
    assert(int(np.sum(orig_idx!=swap_idx)/2)>0)
    num_swaps.append(np.sum(orig_idx!=swap_idx)/2)
    dist_swaps.append(np.abs(orig_idx-swap_idx).sum()/2)
    
    # real
    swap_vecs = extract_tensor(swapped, indices=np.argsort(swap_idx), split_words=use_subwords)
    swap_vecs_zscore = (swap_vecs-m)/s
    
    diff = orig_vecs_zscore-swap_vecs_zscore
    
    orig_centred = orig_vecs - m
    swap_centred = swap_vecs - m
    normalizer = (la.norm(orig_centred,2,1,keepdims=True)*la.norm(swap_centred,2,1,keepdims=True))
    csim.append(np.sum((orig_centred*swap_centred)/normalizer,1))
    
    frob.append(la.norm(diff,'fro',axis=(1,2))/np.sqrt(np.prod(diff.shape[1:])))
    nuc.append(la.norm(diff,'nuc',axis=(1,2))/np.sqrt(np.prod(diff.shape[1:])))
    inf.append(la.norm(diff, np.inf,axis=(1,2))/np.sqrt(np.prod(diff.shape[1:])))
    
    whichline.append(line_idx)
    whichcond.append(these_n[c])
    num_tok.append(ntok)
    
    pbar.update(1)
    
    if np.all(num_cond >= max_num):
        break
    # print('Done with line %d in %.3f seconds'%(i,time()-t0))

fold = 'ngram_swaps/'
if use_subwords:
    fold += 'using_subwords/'
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
np.save(open(SAVE_DIR+fold+'_num_tok.npy','wb'), num_tok)
np.save(open(SAVE_DIR+fold+'_num_swap.npy','wb'),np.array(num_swaps))
np.save(open(SAVE_DIR+fold+'_dist_swap.npy','wb'),np.array(dist_swaps))
# np.save(open(SAVE_DIR+fold+'_average_norms.npy','wb'), np.concatenate(norms,-1))

# print('Done!'



