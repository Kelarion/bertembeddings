
import socket
if socket.gethostname() == 'kelarion':
    CODE_DIR = '/home/matteo/Documents/github/'
    SAVE_DIR = '/home/matteo/Documents/github/bertembeddings/'
else:    
    CODE_DIR = '/rigel/home/ma3811/bert/'
    SAVE_DIR = '/rigel/theory/users/ma3811/bert/'

import sys
# sys.path.append(CODE_DIR+'repler/src/')

sys.path.append(CODE_DIR+'bertembeddings/')
from brackets2trees import BracketedSentence
from pwcca import compute_pwcca
    
from transformers import BertTokenizer, BertModel, BertConfig
import pickle as pkl

import numpy as np
import scipy.linalg as la
import torch

import os
import csv
import random
from time import time
import getopt


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

#%%
allargs = sys.argv

arglist = allargs[1:]

unixOpts = "vl:"
gnuOpts = ["verbose"]

opts, _ = getopt.getopt(arglist, unixOpts, gnuOpts)

verbose, N, init = False, None, None # defaults
for op, val in opts:
    if op in ('-v','--verbose'):
        verbose = True
    if op in ('-l'):
        start = int(val)
        stop = start+100
        
#%%
def rindex(alist, value):
    return len(alist) - alist[-1::-1].index(value) - 1


def extract_tensor(text_array, indices=None, num_layers=13):
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
        bert_output = model(input_ids)[2]
        
    # Index of sorted line
    layer_vectors = []
    for layer in layers:
        list_of_vectors = []
        for word_idx in range(len(text_array)):
            this_word_idx = word_idx + 1

            # the average vector for the subword will be used
            vector_idcs = np.argwhere(np.array(split_word_idx) == this_word_idx).reshape(-1) + 1
            token_vector = bert_output[layer][0][vector_idcs].mean(0).cpu().reshape(-1, 1).numpy()

            list_of_vectors.append(token_vector)
        concatenated_vectors = np.concatenate(list_of_vectors, 1)
        if indices is not None:
            concatenated_vectors = concatenated_vectors[:, indices].reshape(-1, len(indices))
        layer_vectors.append(concatenated_vectors)

    return np.stack(layer_vectors)

#%%
# random_model = False
line_idx = 4

# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# if random_model:
#     model = BertModel(BertConfig(output_hidden_states=True))
#     base_directory = 'vectors/permuted_depth/bert_untrained/'
# else:
#     model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)
#     base_directory = 'extracted/bert-base-cased/'


with open(SAVE_DIR+'data/train_bracketed.txt', 'r') as dfile:
    for i in range(line_idx+1):
        line = dfile.readline()
brak = BracketedSentence(line)



cors = np.zeros((13,8,5,2,100)) # (layer, tree_dist, seq_dist, line)
print('Beginning extraction ... ')
with open(SAVE_DIR+'data/extracted/swapped_data.pkl', 'rb+') as dfile:
    dist = pkl.load(dfile)
# with open(SAVE_DIR+'data/train_bracketed.txt', 'r') as dfile:
    # dist = pkl.load(dfile)
    # t0 = time()
    for i in range(stop):
        if i<start:
            break
        t0 = time()
        
        line = dfile.readline()
        
        sentence = BracketedSentence(line)
        
        # orig_sent = sentence.words
        # d_tree = line[5]
        
        ntok = sentence.ntok
        
        # cc = [[] for _ in range(13)]
        dtmax = np.min([6, ntok])
        # npair = np.sum(ntok - np.arange(1,dtmax))
        cc1 = np.zeros((13, 8, dtmax-1, ntok))
        cc2 = np.zeros((13, 8, dtmax-1, ntok))
        # cca = np.zeros((13, 100, dtmax-1, ntok))*np.nan
        # vecs_original = np.zeros((13, 768, npair, 2))
        # vecs_swapped = np.zeros((13, 768, npair, 2))
        j=0
        for dt in range(1,dtmax):
            swap_idx = (t, t+dt)
            d_tree = sentence.tree_dist(t, t+dt)
            # if d_tree>10:
            #     break
            
            # swapped_sent = list(orig_sent)
            # swapped_sent[swap_idx[0]] = orig_sent[swap_idx[1]]
            # swapped_sent[swap_idx[1]] = orig_sent[swap_idx[0]]
            
            # original_indices = list(range(len(orig_sent)))
    
            # swapped_indices = list(range(len(orig_sent)))
            # swapped_indices[swap_idx[0]] = swap_idx[1]
            # swapped_indices[swap_idx[1]] = swap_idx[0]
            
            # # original sentence
            # original_vectors = extract_tensor(orig_sent, original_indices)
            # # og = original_vectors/la.norm(original_vectors,axis=1)[:,None,:]
            # # og = original_vectors - original_vectors.mean(1)[:,None,:]
            # # swapped sentence
            # swapped_vectors = extract_tensor(swapped_sent, swapped_indices)
            # # sw = swapped_vectors/la.norm(swapped_vectors, axis=1)[:,None,:]
            # # sw = swapped_vectors - swapped_vectors.mean(1)[:,None,:]
            
            # cs_all = np.einsum('...ij,...ik', og[...,swap_idx], sw[...,np.flip(swap_idx)])
            # cs_all /= (np.std(og,axis=1)[:,swap_idx]*np.std(sw,axis=1)[:,np.flip(swap_idx)])[:,:,None]
            for l in range(13):
                C = np.corrcoef(original_vectors[l,:,swap_idx],
                                swapped_vectors[l,:,swap_idx], rowvar=True)
                # d = C[[0,1],[2,3]].mean()
                # d = np.mean(cs_all[l,[0,1],[0,1]])
                # cs[l].append((d,swap_idx))
                # cors[l,d_tree,dt-1,i] += d/(ntok-dt)
                cc1[l, d_tree-2, dt-1, t] = C[0,2]
                cc2[l, d_tree-2, dt-1, t] = C[1,3]
            
            # vecs_original[:,:,j,:] = original_vectors[:,:,swap_idx]
            # vecs_swapped[:,:,j,:] = swapped_vectors[:,:,swap_idx]
            
            j+=1
        
        # pkl.dump(vecs_original, open(SAVE_DIR+'data/line%d_BERT_original.pkl'%i, 'wb+'))
        # pkl.dump(vecs_swapped, open(SAVE_DIR+'data/line%d_BERT_swapped.pkl'%i, 'wb+'))
        cors[:,:,:dtmax-1,0,i-start] = np.nanmean(cc1, axis=-1)
        cors[:,:,:dtmax-1,1,i-start] = np.nanmean(cc2, axis=-1)
        # for i,l in enumerate(cs):
        
    # pkl.dump(cors, open(SAVE_DIR+'data/line%d_swapsimilarity.pkl'%line_idx,'wb+'))
    # pkl.dump(original_vectors, open(directory + 'original_vectors.pkl', 'wb+'))
    # pkl.dump(swapped_vectors, open(directory + 'swapped_vectors.pkl', 'wb+'))
    
        print('Done with line %d, after %.3f seconds'%(i, time()-t0))
        
pkl.dump(cors, open(SAVE_DIR+'data/correlations_line%d-%d.pkl'%(start,stop),'wb+'))











