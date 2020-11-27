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

def extract_tensor(text_array, indices=None, num_layers=12, index_i=0, all_attn=False):
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
        bert_output = model(input_ids)
        full_z = bert_output[2]
        output_att = bert_output[3]
        output_Z = bert_output[4]
        
    # output_Z: layer, head, nTok, 64
    # output_att: layer, head, nTok, nTok
    # emb: layer, nTok, 768
    layer_full_vectors = []
    layer_Z_vectors = []
    layer_att = []
    for layer in layers:
        this_layer_full_z = []
        this_layer_z = []
        this_layer_attn = []
        for word_idx in range(len(text_array)):
            this_word_idx = word_idx + 1
            
            list_of_Z_vectors = []
            list_of_att_weights = []
            
            vector_idcs = np.argwhere(np.array(split_word_idx) == this_word_idx).reshape(-1) + 1
            token_vector = full_z[layer][0][vector_idcs].mean(0).cpu().reshape(-1, 1).numpy()
            this_layer_full_z.append(token_vector)
            
            for head in range(12):
                Z_vector = output_Z[layer][0][head][vector_idcs].mean(0).cpu().reshape(-1, 1).numpy()
                
                if all_attn:
                    attn_vectors = output_att[layer][0][head][vector_idcs,:].mean(0)
                    # need to take two average for attention
                    attn_mean = []
                    for word_idx in range(len(text_array)):
                        this_word_idx = word_idx + 1
                        vector_idcs = np.argwhere(np.array(split_word_idx) == this_word_idx).reshape(-1) + 1
                        attn_mean.append(attn_vectors[vector_idcs].mean())
                    att_weight = np.array(attn_mean).T
                    list_of_att_weights.append(att_weight)
                else:
                    if index_i == word_idx:
                        vector_idcs_next = np.argwhere(np.array(split_word_idx) ==
                                                       this_word_idx+1).reshape(-1) + 1
                        att_weight = np.sum([np.average([output_att[layer][0][head][i][j] for i in vector_idcs]) for j in
                                             vector_idcs_next])  # Att(w1->w2)
                        list_of_att_weights.append(att_weight)
                    elif index_i+1 == word_idx:
                        vector_idcs_prev = np.argwhere(np.array(split_word_idx) ==
                                                       this_word_idx - 1).reshape(-1) + 1
                        att_weight = np.sum(
                            [np.average([output_att[layer][0][head][i][j] for i in vector_idcs]) for j
                             in
                             vector_idcs_prev])# Att(w2->w1)
                        list_of_att_weights.append(att_weight)
                list_of_Z_vectors.append(Z_vector)
                
            concatenated_Z_vectors = np.concatenate(list_of_Z_vectors, 1).T
            this_layer_z.append(concatenated_Z_vectors)
            if len(list_of_att_weights)>0:
                this_layer_attn.append(list_of_att_weights)
        
        cat_layer_z = np.stack(this_layer_z)
        cat_layer_full = np.stack(this_layer_full_z)
        if indices is not None:
            cat_layer_z = cat_layer_z[indices, ...]
            cat_layer_full = cat_layer_full[indices, ...]
            
        layer_full_vectors.append(cat_layer_full.transpose((1,0,2)).squeeze())
        layer_Z_vectors.append(cat_layer_z.transpose((1,0,2)))
        layer_att.append(this_layer_attn)
        
    # full_vecs = np.array(layer_full_vectors).squeeze().transpose((0,2,1))
    return np.stack(layer_Z_vectors), np.array(layer_att), np.array(layer_full_vectors)

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


dfile = SAVE_DIR+'train_bracketed.txt'

# with open(SAVE_DIR+'permuted_data.pkl', 'rb') as dfile:
#     dist = pkl.load(dfile)

#%%
max_num = 200
these_bounds = [0,1,2,3,4,5,6]

frob = []
frob_full = []
nuc = []
inf = []
csim = []
whichline = []
whichcond = []
whichswap = []
attn = []
attn_orig = []
concentration = []
norms = []
dist_avg = []

print('Computing mean and variance ...')
all_vecs = []
all_full_vecs = []
for line_idx in np.random.choice(range(5000), 300):
    
    line = linecache.getline(dfile, line_idx+1)
    sentence = BracketedSentence(line)
    if sentence.ntok<10:
        continue
    
    orig = sentence.words
    ntok = sentence.ntok
    
    orig_idx = np.arange(ntok)
    
    swap_idx = np.random.permutation(orig_idx)
    swapped = [orig[i] for i in swap_idx]
    
    orig_vecs, _, orig_full = extract_tensor(orig, indices=orig_idx)
    swap_vecs, _, swap_full = extract_tensor(swapped, indices=swap_idx)
    
    catted = np.append(orig_vecs, swap_vecs, -2)
    all_vecs.append(catted)
    all_full_vecs.append(np.append(orig_full, swap_full, -1))
    
m = np.concatenate(all_vecs,-2).mean(-2,keepdims=True)
s = np.concatenate(all_vecs,-2).std(-2,keepdims=True)

m_full = np.concatenate(all_full_vecs,-1).mean(-1,keepdims=True)
s_full = np.concatenate(all_full_vecs,-1).std(-1,keepdims=True)

# remember:
# output[0] is activations: layer, head, nTok, 64
# output[1] is attentions: layer, head, nTok, nTok

num_cond = np.zeros(len(these_bounds))
t0 = time()
pbar = tqdm.tqdm(total=max_num*len(these_bounds))
for line_idx in np.random.permutation(range(5000)[:1000]):
    
    line = linecache.getline(dfile, line_idx+1)
    sentence = BracketedSentence(line)
    if sentence.ntok<10:
        continue
    # orig = d[0]
    orig = sentence.words
    ntok = sentence.ntok
    
    crossings = np.diff(np.abs(sentence.brackets).cumsum()[sentence.term2brak])
    if not np.any(np.isin(crossings, these_bounds)):
        continue
    
    orig_idx = np.array(range(ntok))
    for i,c in enumerate(crossings):
        if (c not in these_bounds) or (num_cond[c] >= max_num):
            continue
        num_cond[c] += 1
        
        swap_idx = np.array(range(ntok))
        swap_idx[i+1] = i
        swap_idx[i] = i+1
        
        swapped = [orig[i] for i in swap_idx]
        
        # real
        orig_vecs, orig_attn, orig_full = extract_tensor(orig, indices=orig_idx, index_i=i) # (Layer, Head, 64, Tok)
        swap_vecs, swap_attn, swap_full = extract_tensor(swapped, indices=swap_idx, index_i=i)
        
        attn.append(np.sum(swap_attn,-1)/2)
        attn_orig.append(np.sum(orig_attn,-1)/2)
        
        orig_vecs_zscore = (orig_vecs-m)/s
        swap_vecs_zscore = (swap_vecs-m)/s
    
        # orig_centred = orig_vecs-orig_vecs.mean(axis=3, keepdims=True)
        # swap_centred = swap_vecs-swap_vecs.mean(axis=3, keepdims=True)
        normalizer = (la.norm(orig_vecs,2,2,keepdims=True)*la.norm(swap_vecs,2,2,keepdims=True))
        # csim.append(np.sum((orig_centred*swap_centred)/normalizer,2))
        csim.append(np.sum((orig_vecs*swap_vecs)/normalizer,axis=(2,3)))
        
        diff = orig_vecs_zscore-swap_vecs_zscore
        diff_full = (orig_full-swap_full)/s_full
        frob.append(la.norm(diff,'fro', axis=(2,3))/np.sqrt(np.prod(diff.shape[1:])))
        frob_full.append(la.norm(diff_full,'fro', axis=(1,2))/np.sqrt(np.prod(diff_full.shape[1:])))
        
        nuc.append(la.norm(diff,'nuc', axis=(2,3))/np.sqrt(np.prod(diff.shape[1:])))
        inf.append(la.norm(diff, np.inf, axis=(2,3))/np.sqrt(np.prod(diff.shape[1:])))
        dist_avg.append(la.norm(diff, 2, axis=2).mean(2))
        
        norms.append(la.norm(np.append(orig_vecs-m, swap_vecs-m, -1), 2, 2).mean(-1))
        
        whichline.append(line_idx)
        whichcond.append(c)
        whichswap.append(np.repeat(len(whichline), ntok))
        
        pbar.update(1)
    
    if np.all(num_cond >= max_num):
        break
    # print('Done with line %d in %.3f seconds'%(i,time()-t0))

fold = 'bracket_crossings/'

if random_model:
    fold += 'random_model/'
    

if not os.path.isdir(SAVE_DIR+fold):
    os.makedirs(SAVE_DIR+fold)

# pref = '%s_%dorder'%(phrase_type, order)

np.save(open(SAVE_DIR+fold+'_cosines.npy','wb'),np.stack(csim))
np.save(open(SAVE_DIR+fold+'_frob.npy','wb'),np.stack(frob))
np.save(open(SAVE_DIR+fold+'_frob_full.npy','wb'),np.stack(frob_full))
np.save(open(SAVE_DIR+fold+'_nuc.npy','wb'),np.stack(nuc))
np.save(open(SAVE_DIR+fold+'_inf.npy','wb'),np.stack(inf))
np.save(open(SAVE_DIR+fold+'_line_id.npy','wb'),whichline)
np.save(open(SAVE_DIR+fold+'_num_crossings.npy','wb'), whichcond)
np.save(open(SAVE_DIR+fold+'_swap_id.npy','wb'), np.concatenate(whichswap))
np.save(open(SAVE_DIR+fold+'_attn_swap.npy','wb'), np.stack(attn))
np.save(open(SAVE_DIR+fold+'_attn_orig.npy','wb'), np.stack(attn_orig))
np.save(open(SAVE_DIR+fold+'_average_norms.npy','wb'), np.stack(norms))
np.save(open(SAVE_DIR+fold+'_dist_avg.npy','wb'), np.stack(dist_avg))


print('Done!')


# to do: the controlled regression on each head -- which ones are significant?

