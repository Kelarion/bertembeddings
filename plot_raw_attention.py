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
            list_of_att_weights.append(output_att[layer][0][head].numpy())
            for word_idx in range(len(text_array)):
                this_word_idx = word_idx + 1
                vector_idcs = np.argwhere(np.array(split_word_idx) == this_word_idx).reshape(-1) + 1
                Z_vector = output_Z[layer][0][head][vector_idcs].mean(0).cpu().reshape(-1, 1).numpy()
                
                if all_attn:
                    # attn_vectors = output_att[layer][0][head][:,vector_idcs].sum(1)
                    # # need to take two average for attention
                    # attn_mean = []
                    # for word_idx in range(len(text_array)):
                    #     this_word_idx = word_idx + 1
                    #     vector_idcs = np.argwhere(np.array(split_word_idx) == this_word_idx).reshape(-1) + 1
                    #     attn_mean.append(attn_vectors[vector_idcs].mean())
                    # att_weight = np.array(attn_mean).T
                    # list_of_att_weights.append(att_weight)
                    # list_of_att_weights.append(output_att[layer][0][head].numpy())
                    ''
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
        
    return np.stack(layer_Z_vectors), np.array(layer_att).squeeze()[:,:,1:-1,1:-1], np.array(split_word_idx)

#%%
model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

dfile = SAVE_DIR+'train_bracketed.txt'

#%%
line_idx = 4

line = linecache.getline(dfile, line_idx+1)
sentence = BracketedSentence(line)
orig = sentence.words
ntok = sentence.ntok
orig_idx = np.arange(ntok)

# output_att: layer, head, nTok, nTok
orig_vecs, orig_attn, word_idx = extract_tensor(orig, indices=orig_idx, all_attn=True)

#%%
phrase_order = 1
# layers = list(range(12))
layers = [9]

phr = sentence.phrases(phrase_order)
bounds = [0] + [word_idx.tolist().index(p[-1]) for p in phr]

plt.figure()
for j,l in enumerate(layers):
    for h in range(12):
        n = len(layers)*j + h + 1
        plt.subplot(len(layers),12,n)
        plt.imshow(orig_attn[l,h,:,:],'gray')
        
        for i,p in enumerate(phr):
            x = [p[0]-0.5,p[0]-0.5,p[-1]+0.5,p[-1]+0.5,p[0]-0.5]
            y = [p[0]-0.5,p[-1]+0.5,p[-1]+0.5,p[0]-0.5,p[0]-0.5]
            plt.plot(x,y,'w')
        
        plt.xticks([])
        plt.yticks([])
        # plt.clim([0,1])
        
#%%

concentration = []
for line_idx in tqdm.tqdm(np.random.choice(range(5000), 300)):
    
    line = linecache.getline(dfile, line_idx+1)
    sentence = BracketedSentence(line)
    if sentence.ntok<10:
        continue
    
    orig = sentence.words
    ntok = sentence.ntok
    
    
    orig_vecs, orig_attn, word_idx = extract_tensor(orig, all_attn=True)
    
    attn = orig_attn[:,:,1:-1,1:-1]
    
    didx = np.abs(word_idx[:,None]-word_idx[None,:])
    conc = (orig_attn*didx).sum(-1).mean(-1)
    
    concentration.append(conc)


        
        
        