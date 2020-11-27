SAVE_DIR = 'C:/Users/mmall/Documents/github/bertembeddings/data/'
CODE_DIR = 'C:/Users/mmall/Documents/github/bertembeddings/'

import sys, os
sys.path.append(CODE_DIR)
from brackets2trees import BracketedSentence

import torch
import torch.nn as nn
import torch.optim as optim

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
# # random_model = True
random_model = False

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
    
dfile = SAVE_DIR+'train_bracketed.txt'

pos_tags = list(np.load(SAVE_DIR+'unique_pos_tags.npy'))
phrase_tags = list(np.load(SAVE_DIR+'unique_phrase_tags.npy'))

#%%
bsz = 20
nepoch = 10
lr = 1e-3

glm_orig_pos = nn.Linear(768, len(pos_tags), bias=False)
glm_swap_pos = nn.Linear(768, len(pos_tags), bias=False)
glm_orig_syn = nn.Linear(768, len(phrase_tags), bias=False)
glm_swap_syn = nn.Linear(768, len(phrase_tags), bias=False)

layer = 12

optimizer_orig_pos = optim.Adam(glm_orig_pos.parameters(),lr)
optimizer_swap_pos = optim.Adam(glm_swap_pos.parameters(),lr)
optimizer_orig_syn = optim.Adam(glm_orig_syn.parameters(),lr)
optimizer_swap_syn = optim.Adam(glm_swap_syn.parameters(),lr)

swap_idx_all = [[] for _ in range(5000)]

train_loss_orig_pos = []
train_loss_swap_pos = []
train_loss_orig_syn = []
train_loss_swap_syn = []
for epoch in range(nepoch):
    nbatch = 0  # we'll only take gradient steps every bsz datapoints
    cumloss_orig_pos = 0
    cumloss_swap_pos = 0
    cumloss_orig_syn = 0
    cumloss_swap_syn = 0
    optimizer_orig_pos.zero_grad()
    optimizer_swap_pos.zero_grad()
    optimizer_orig_syn.zero_grad()
    optimizer_swap_syn.zero_grad()

    num_batch = int(5000/bsz)
    with tqdm.tqdm(range(num_batch),total=num_batch, desc='Epoch %d'%epoch, postfix=[dict(loss_orig_pos_=0,
                                                                                          loss_swap_pos_=0,
                                                                                          loss_orig_syn_=0,
                                                                                          loss_swap_syn_=0)]) as pbar:
        for line_idx in np.random.permutation(range(5000)):
            
            line = linecache.getline(dfile, line_idx+1)
            sentence = BracketedSentence(line)
            
            orig = sentence.words
            ntok = sentence.ntok
            
            orig_idx = np.array(range(ntok))
            if len(swap_idx_all[line_idx]) == 0:
                swap_idx_all[line_idx] = np.random.permutation(orig_idx)
            swap_idx = swap_idx_all[line_idx]
            
            orig_vecs = extract_tensor(orig, indices=orig_idx)
            swap_vecs = extract_tensor([orig[s] for s in swap_idx], indices=swap_idx)
            
            labels_pos = np.array([pos_tags.index(w) if w in pos_tags else np.nan for w in sentence.pos_tags])
            keep = ~np.isnan(labels_pos)
            y_pos = torch.tensor(labels_pos[keep]).long()
            X_orig_pos = torch.tensor(orig_vecs[layer,:,keep])
            X_swap_pos = torch.tensor(swap_vecs[layer,:,keep])
            
            anc_tags = [sentence.ancestor_tags(i, 2) for i in range(sentence.ntok)]
            labels_syn = np.array([phrase_tags.index(w) if w in phrase_tags else np.nan for w in anc_tags])
            keep = ~np.isnan(labels_syn)
            y_syn = torch.tensor(labels_syn[keep]).long()
            X_orig_syn = torch.tensor(orig_vecs[layer,:,keep])
            X_swap_syn = torch.tensor(swap_vecs[layer,:,keep])
            
            
            loss_orig_pos = nn.CrossEntropyLoss()(glm_orig_pos(X_orig_pos), y_pos)
            loss_swap_pos = nn.CrossEntropyLoss()(glm_swap_pos(X_swap_pos), y_pos)
            loss_orig_syn = nn.CrossEntropyLoss()(glm_orig_syn(X_orig_syn), y_syn)
            loss_swap_syn = nn.CrossEntropyLoss()(glm_swap_syn(X_swap_syn), y_syn)
            
            # running_loss_swap_syn += loss.item()
            
            if nbatch<bsz: # still in batch
                cumloss_orig_pos += loss_orig_pos
                cumloss_swap_pos += loss_swap_pos
                cumloss_orig_syn += loss_orig_syn
                cumloss_swap_syn += loss_swap_syn
                nbatch+=1
            else: # end of batch
                train_loss_orig_pos.append(cumloss_orig_pos.item())
                train_loss_swap_pos.append(cumloss_swap_pos.item())
                train_loss_orig_syn.append(cumloss_orig_syn.item())
                train_loss_swap_syn.append(cumloss_swap_syn.item())
                cumloss_orig_pos.backward()
                cumloss_swap_pos.backward()
                cumloss_orig_syn.backward()
                cumloss_swap_syn.backward()
                optimizer_orig_pos.step()
                optimizer_swap_pos.step()
                optimizer_orig_syn.step()
                optimizer_swap_syn.step()
                
                pbar.postfix[0]['loss_orig_pos_'] = np.round(cumloss_orig_pos.item(), 3)
                pbar.postfix[0]['loss_swap_pos_'] = np.round(cumloss_swap_pos.item(), 3)
                pbar.postfix[0]['loss_orig_syn_'] = np.round(cumloss_orig_syn.item(), 3)
                pbar.postfix[0]['loss_swap_syn_'] = np.round(cumloss_swap_syn.item(), 3)
                pbar.update()
                
                cumloss_orig_pos = 0
                cumloss_swap_pos = 0
                cumloss_orig_syn = 0
                cumloss_swap_syn = 0
                nbatch = 0
                optimizer_orig_pos.zero_grad()
                optimizer_swap_pos.zero_grad()
                optimizer_orig_syn.zero_grad()
                optimizer_swap_syn.zero_grad()

#%%
fold = 'linear_probes/'
if random_model:
    fold += 'random_model/'

if not os.path.isdir(SAVE_DIR+fold):
    os.makedirs(SAVE_DIR+fold)
    
np.save(SAVE_DIR+fold+'orig_pos_layer%d_train_loss.npy'%layer, train_loss_orig_pos)
np.save(SAVE_DIR+fold+'swap_pos_layer%d_train_loss.npy'%layer, train_loss_swap_pos)
np.save(SAVE_DIR+fold+'orig_syn_layer%d_train_loss.npy'%layer, train_loss_orig_syn)
np.save(SAVE_DIR+fold+'swap_syn_layer%d_train_loss.npy'%layer, train_loss_swap_syn)
with open(SAVE_DIR+fold+'orig_pos_layer%d_params.pt'%layer, 'wb') as f:
    torch.save(glm_orig_pos.state_dict(), f)
with open(SAVE_DIR+fold+'swap_pos_layer%d_params.pt'%layer, 'wb') as f:
    torch.save(glm_swap_pos.state_dict(), f)
with open(SAVE_DIR+fold+'orig_syn_layer%d_params.pt'%layer, 'wb') as f:
    torch.save(glm_orig_syn.state_dict(), f)
with open(SAVE_DIR+fold+'swap_syn_layer%d_params.pt'%layer, 'wb') as f:
    torch.save(glm_swap_syn.state_dict(), f)
