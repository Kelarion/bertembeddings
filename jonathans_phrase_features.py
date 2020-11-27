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

#%% for sanity checking
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
jon_folder = 'C:/Users/mmall/Documents/github/bertembeddings/data/jonathans/'
random_model = False
# random_model = True

if random_model:
    model_dir = '/phrase/bert-base-cased_untrained/'
    model = BertModel(BertConfig(output_hidden_states=True, output_attentions=True))
else:
    model_dir = '/ngram/bert-base-cased/'
    model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

lines = os.listdir(jon_folder+model_dir)
dist = pkl.load(open(jon_folder+'phrase/permuted_data.pkl','rb'))

dfile = SAVE_DIR+'train_bracketed.txt'

#%%
extract_myself = True
# extract_myself = False

frob = []
frob_unswp = []
frob_swp = []
nuc = []
inf = []
csim = []
avgdist = []
whichline = []
whichcond = []
whichswap = []
norms = []
# mean = np.zeros((13, 768))

# compute mean and variance for z-scoring
print('Computing mean and variance ...')
all_vecs = []
for line in tqdm.tqdm(lines):
    
    info = dist[int(line)]
    
    if extract_myself:
        orig = info[0]
        orig_idx = range(len(orig))
        
        swp_idx = info[2][:2]
        splt_idx = np.concatenate([(s[0],s[1]+1) for s in swp_idx])
        chunked = np.split(np.arange(len(orig)),splt_idx)
        swap_idx = np.concatenate(np.array(chunked)[[0,3,2,1,4]])
        swapped = [orig[i] for i in swap_idx]    
        assert(swapped == info[2][2])
        
        orig_vecs = extract_tensor(orig)
        swap_vecs = extract_tensor(swapped, indices=np.argsort(swap_idx))
    else:
        orig_vecs = pkl.load(open(jon_folder+model_dir+line+'/normal_vectors.pkl','rb'))
        swap_vecs = pkl.load(open(jon_folder+model_dir+line+'/fake_vectors.pkl','rb'))

    catted = np.append(orig_vecs, swap_vecs, -1)
    # means.append(catted.mean(-1))
    # var.append(((catted-catted.mean(-1,keepdims=True))**2).mean(-1))
    all_vecs.append(catted)
m = np.concatenate(all_vecs,-1).mean(-1,keepdims=True)
s = np.concatenate(all_vecs,-1).std(-1,keepdims=True)


for line_idx, line in enumerate(tqdm.tqdm(lines)):
    
    info = dist[int(line)]
    if extract_myself:
        orig = info[0]
        orig_idx = range(len(orig))
        
        orig_vecs = extract_tensor(orig)
    else:
        orig_vecs = pkl.load(open(jon_folder+model_dir+line+'/normal_vectors.pkl','rb'))
    orig_vecs_zscore = (orig_vecs-m)/s
    
    ntok = orig_vecs.shape[-1]
    
    for p, phrase_type in enumerate(['real','fake']): # [real, imitation] phrase swaps
        
        if extract_myself:
            swp_idx = info[p+1][:2]
            splt_idx = np.concatenate([(s[0],s[1]+1) for s in swp_idx])
            chunked = np.split(np.arange(len(orig)),splt_idx)
            swap_idx = np.concatenate(np.array(chunked)[[0,3,2,1,4]])
            swapped = [orig[i] for i in swap_idx]
            assert(swapped == info[p+1][2])
            
            swap_vecs = extract_tensor(swapped, indices=np.argsort(swap_idx))
        else:
            swap_vecs = pkl.load(open(jon_folder+model_dir+line+'/%s_vectors.pkl'%phrase_type,'rb'))
        
        swp_words = list(range(info[1+p][0][0],info[1+p][0][1]+1)) + list(range(info[1+p][1][0],info[1+p][1][1]+1))
        is_swp = np.isin(list(range(ntok)), swp_words)
        
        orig_centred = orig_vecs - m
        swap_centred = swap_vecs - m
        normalizer = (la.norm(orig_centred,2,1,keepdims=True)*la.norm(swap_centred,2,1,keepdims=True))
        csim.append(np.sum((orig_centred*swap_centred)/normalizer,1))
        
        swap_vecs_zscore = (swap_vecs-m)/s
        # orig_vecs_zscore = orig_vecs
        # swap_vecs_zscore = swap_vecs
        
        diff = orig_vecs_zscore-swap_vecs_zscore
        
        frob.append(la.norm(diff, 'fro', axis=(1,2))/np.sqrt(ntok)/np.sqrt(768))
        frob_swp.append(la.norm(diff[:,:,is_swp], 'fro', axis=(1,2))/np.sqrt(is_swp.sum())/np.sqrt(768))
        frob_unswp.append(la.norm(diff[:,:,~is_swp], 'fro', axis=(1,2))/np.sqrt((~is_swp).sum())/np.sqrt(768))
        nuc.append(la.norm(diff, 'nuc', axis=(1,2))/np.sqrt(ntok))
        inf.append(la.norm(diff, np.inf, axis=(1,2))/np.sqrt(ntok))
        avgdist.append(la.norm(diff, 2, axis=1).mean(1))
        
        norms.append(la.norm(np.append(orig_vecs, swap_vecs, -1), 2, axis=1))
        # norms.append(np.append(orig_vecs, swap_vecs, -1).sum(-1))
        
        whichline.append(line_idx)
        whichcond.append(p)
        whichswap.append(np.repeat(len(whichline), ntok))
        
    # if np.all(num_cond >= max_num):
    #     break
    # print('Done with line %d in %.3f seconds'%(i,time()-t0))

fold = 'phrase_swaps/jonathans/'
if random_model:
    fold += 'random_model/'

if not os.path.isdir(SAVE_DIR+fold):
    os.makedirs(SAVE_DIR+fold)

# pref = '%s_%dorder'%(phrase_type, order)

# norms = np.concatenate(norms,-1)

np.save(open(SAVE_DIR+fold+'_cosines.npy','wb'),np.concatenate(csim,axis=1))
np.save(open(SAVE_DIR+fold+'_frob.npy','wb'),np.stack(frob))
np.save(open(SAVE_DIR+fold+'_frob_swp.npy','wb'),np.stack(frob_swp))
np.save(open(SAVE_DIR+fold+'_frob_unswp.npy','wb'),np.stack(frob_unswp))
np.save(open(SAVE_DIR+fold+'_nuc.npy','wb'),np.stack(nuc))
np.save(open(SAVE_DIR+fold+'_inf.npy','wb'),np.stack(inf))
np.save(open(SAVE_DIR+fold+'_line_id.npy','wb'),whichline)
np.save(open(SAVE_DIR+fold+'_condition.npy','wb'), whichcond)
np.save(open(SAVE_DIR+fold+'_swap_id.npy','wb'), np.concatenate(whichswap))
np.save(open(SAVE_DIR+fold+'_average_norms.npy','wb'), np.concatenate(norms,-1))
# np.save(open(SAVE_DIR+fold+'_average_norms.npy','wb'), norms)
np.save(open(SAVE_DIR+fold+'_dist_avg.npy','wb'), np.stack(avgdist))
