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
        output_att = bert_output[3]
        output_Z = bert_output[4]
        
    # output_Z: layer, head, nTok, 64
    # output_att: layer, head, nTok, nTok
    # emb: layer, nTok, 768
    layer_Z_vectors = []
    layer_att = []
    for layer in layers:
        this_layer_z = []
        this_layer_attn = []
        for head in range(12):
            list_of_Z_vectors = []
            list_of_att_weights = []
            for word_idx in range(len(text_array)):
                this_word_idx = word_idx + 1
                vector_idcs = np.argwhere(np.array(split_word_idx) == this_word_idx).reshape(-1) + 1
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
                
            concatenated_Z_vectors = np.concatenate(list_of_Z_vectors, 1)
            if indices is not None:
                concatenated_Z_vectors = concatenated_Z_vectors[:, indices].reshape(-1, len(indices))
            this_layer_z.append(concatenated_Z_vectors)
            this_layer_attn.append(list_of_att_weights)
        layer_Z_vectors.append(this_layer_z)
        layer_att.append(this_layer_attn)
            
    return np.stack(layer_Z_vectors), np.array(layer_att)

#%%
model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


dfile = SAVE_DIR+'train_bracketed.txt'

# with open(SAVE_DIR+'permuted_data.pkl', 'rb') as dfile:
#     dist = pkl.load(dfile)

#%%
attn = []
# frob = []

# print('Computing mean and variance ...')
# all_vecs = []
# for line_idx in np.random.choice(range(5000), 300):
    
#     line = linecache.getline(dfile, line_idx+1)
#     sentence = BracketedSentence(line)
#     if sentence.ntok<10:
#         continue
    
#     orig = sentence.words
#     ntok = sentence.ntok
    
#     orig_idx = np.arange(ntok)
    
#     swap_idx = np.random.permutation(orig_idx)
#     swapped = [orig[i] for i in swap_idx]
    
#     orig_vecs = extract_tensor(orig, indices=orig_idx)[0]
#     swap_vecs = extract_tensor(swapped, indices=swap_idx)[0]
    
#     catted = np.append(orig_vecs, swap_vecs, -1)
#     all_vecs.append(catted)
    
# m = np.concatenate(all_vecs,-1).mean(-1,keepdims=True)
# s = np.concatenate(all_vecs,-1).std(-1,keepdims=True)

# remember:
# output[0] is activations: layer, head, nTok, 64
# output[1] is attentions: layer, head, nTok, nTok

for line_idx in np.random.choice(5000,500,replace=False):
    
    line = linecache.getline(dfile, line_idx+1)
    sentence = BracketedSentence(line)
    if sentence.ntok<10:
        continue
    # orig = d[0]
    orig = sentence.words
    ntok = sentence.ntok

    # real
    _, orig_attn = extract_tensor(orig, all_attn=True) # (Layer, Head, 64, Tok)
    
    all_attn = [[orig_attn[i,j,:,:].flatten() for j in range(12)] for i in range(12)]
    
    attn.append(all_attn)
    
    # print('Done with line %d in %.3f seconds'%(i,time()-t0))

fold = 'bracket_crossings/'

if not os.path.isdir(SAVE_DIR+fold):
    os.makedirs(SAVE_DIR+fold)

# pref = '%s_%dorder'%(phrase_type, order)

np.save(open(SAVE_DIR+fold+'_cosines.npy','wb'),np.stack(csim))
np.save(open(SAVE_DIR+fold+'_frob.npy','wb'),np.stack(frob))
np.save(open(SAVE_DIR+fold+'_nuc.npy','wb'),np.stack(nuc))
np.save(open(SAVE_DIR+fold+'_inf.npy','wb'),np.stack(inf))
np.save(open(SAVE_DIR+fold+'_line_id.npy','wb'),whichline)
np.save(open(SAVE_DIR+fold+'_num_crossings.npy','wb'), whichcond)
np.save(open(SAVE_DIR+fold+'_swap_id.npy','wb'), np.concatenate(whichswap))
np.save(open(SAVE_DIR+fold+'_attn.npy','wb'), np.stack(attn))
np.save(open(SAVE_DIR+fold+'_attn_orig.npy','wb'), np.stack(attn_orig))
np.save(open(SAVE_DIR+fold+'_average_norms.npy','wb'), np.stack(norms))
np.save(open(SAVE_DIR+fold+'_dist_avg.npy','wb'), np.stack(dist_avg))


print('Done!')


# to do: the controlled regression on each head -- which ones are significant?

