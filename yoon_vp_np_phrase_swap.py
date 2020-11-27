SAVE_DIR = 'C:/Users/mmall/Documents/github/bertembeddings/data/'
CODE_DIR = 'C:/Users/mmall/Documents/github/bertembeddings/'

import sys, os
sys.path.append(CODE_DIR)
from brackets2trees import BracketedSentence

import torch
import tqdm

from transformers import BertTokenizer, BertModel, BertConfig
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
            print(vector_idcs.shape)
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

dist = pkl.load(open('C:/Users/mmall/Documents/github/bertembeddings/data/phrase_swaps/permuted_data_np_vp.pkl','rb'))

#%%
include_random = True
# include_random = False
num_lines = 1000

# use_subwords = False
use_subwords = True


these_lines = np.random.choice(len(dist),num_lines,replace=False)
# print('Computing mean and variance ...')
all_vecs = []
for line_idx in tqdm.tqdm(these_lines):
    line = dist[line_idx]
    ntok = len(line[0])
   
    if ntok<10:
        continue
    # orig = d[0]
    orig = line[0]
        
    orig_idx = np.array(range(ntok))
    
    phrase_type = np.random.choice(2)
    swap_type = np.random.choice(1+include_random)
    
    swp_idx = line[1+phrase_type][swap_type][:2]
    splt_idx = np.concatenate([(s[0],s[1]+1) for s in swp_idx])
    chunked = np.split(np.arange(ntok),splt_idx)
    swap_idx = np.concatenate(np.array(chunked)[[0,3,2,1,4]])
    
    swapped = [orig[i] for i in swap_idx]
    
    # assert(swapped == line[1+phrase_type][swap_type][2])
    assert([swapped[i] for i in np.argsort(swap_idx)] == orig)
    
    # real
    orig_vecs = extract_tensor(orig, split_words=use_subwords)
    swap_vecs = extract_tensor(swapped, indices=np.argsort(swap_idx), split_words=use_subwords)
    
    all_vecs.append(np.append(orig_vecs, swap_vecs, -1))
m = np.concatenate(all_vecs,-1).mean(-1,keepdims=True)
s = np.concatenate(all_vecs,-1).std(-1,keepdims=True)


num_swaps = [] # track the number of words swapped in each sentence
dist_swaps = []
num_tok = []
frob = []
nuc = []
inf = []
csim = []
whichline = []
whichcond = []

t0 = time()
for line_idx in tqdm.tqdm(these_lines):
    
    line = dist[line_idx]
    
    orig = line[0]
    ntok = len(orig)
    
    orig_idx = np.array(range(ntok))
    
    orig_vecs = extract_tensor(orig, split_words=use_subwords)
    orig_vecs_zscore = (orig_vecs-m)/s
    
    if include_random:
        remaining = np.array(line[1:]).reshape(4,-1).tolist()
    else:
        remaining = [l[0] for l in line[1:]]
    
    for phrase_type, swap_line in enumerate(remaining):
        
        swp_idx = swap_line[:2]
        splt_idx = np.concatenate([(s[0],s[1]+1) for s in swp_idx])
        chunked = np.split(np.arange(ntok),splt_idx)
        swap_idx = np.concatenate(np.array(chunked)[[0,3,2,1,4]])
        
        swapped = [orig[i] for i in swap_idx]    
        num_swaps.append(sum(orig_idx!=swap_idx)/2)
        dist_swaps.append(np.abs(orig_idx-swap_idx).sum()/2)
        assert(int(np.sum(orig_idx!=swap_idx)/2)>0)
        
        swap_vecs = extract_tensor(swapped, indices=np.argsort(swap_idx), split_words=use_subwords)
        
        swap_vecs_zscore = (swap_vecs-m)/s
        
        diff = orig_vecs_zscore-swap_vecs_zscore
        
        orig_centred = orig_vecs-orig_vecs.mean(axis=2, keepdims=True)
        swap_centred = swap_vecs-swap_vecs.mean(axis=2, keepdims=True)
        normalizer = (la.norm(orig_centred,2,1,keepdims=True)*la.norm(swap_centred,2,1,keepdims=True))
        csim.append(np.sum((orig_centred*swap_centred)/normalizer,1))
        
        frob.append(la.norm(diff,'fro',axis=(1,2))/np.sqrt(np.prod(diff.shape[1:])))
        nuc.append(la.norm(diff,'nuc',axis=(1,2))/np.sqrt(np.prod(diff.shape[1:])))
        inf.append(la.norm(diff, np.inf,axis=(1,2))/np.sqrt(np.prod(diff.shape[1:])))
        # avgdist.append(la.norm(diff, 2, axis=1).mean(1))
        
        whichline.append(line_idx)
        whichcond.append(phrase_type)
        num_tok.append(ntok)
        
        # print('Done with line %d in %.3f seconds'%(i,time()-t0))

fold = 'phrase_swaps/vpnp/'
if use_subwords:
    fold += 'using_subwords/'
if random_model:
    fold += 'random_model/'

if not os.path.isdir(SAVE_DIR+fold):
    os.makedirs(SAVE_DIR+fold)

pref = ''
# pref = '%s_%dorder'%(phrase_type, order)
# if num_phrases is not None:
#     pref += '_%dphrases'%(num_phrases)
np.save(open(SAVE_DIR+fold+pref+'_cosines.npy','wb'),np.concatenate(csim,axis=1))
np.save(open(SAVE_DIR+fold+pref+'_frob.npy','wb'),np.stack(frob))
np.save(open(SAVE_DIR+fold+pref+'_nuc.npy','wb'),np.stack(nuc))
np.save(open(SAVE_DIR+fold+pref+'_inf.npy','wb'),np.stack(inf))
np.save(open(SAVE_DIR+fold+pref+'_num_tok.npy','wb'),num_tok)
np.save(open(SAVE_DIR+fold+pref+'_num_swap.npy','wb'),num_swaps)
np.save(open(SAVE_DIR+fold+pref+'_dist_swap.npy','wb'),dist_swaps)
np.save(open(SAVE_DIR+fold+pref+'_line_id.npy','wb'),whichline)
np.save(open(SAVE_DIR+fold+pref+'_condition.npy','wb'), whichcond)

print('Done!')

