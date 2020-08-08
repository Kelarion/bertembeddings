SAVE_DIR = '/home/matteo/Documents/github/bertembeddings/data/'
CODE_DIR = '/home/matteo/Documents/github/bertembeddings/'

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
        _, _, bert_output, attn_weight = model(input_ids)
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

model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


dfile = SAVE_DIR+'train_bracketed.txt'

# with open(SAVE_DIR+'permuted_data.pkl', 'rb') as dfile:
#     dist = pkl.load(dfile)

#%%
order = 2
# num_phrases = None
num_phrases = None

swap_type = 'within'
# swap_type = 'among'

for phrase_type in ['real','imitation']:
    num_swaps = [] # track the number of words swapped in each sentence
    frob = []
    nuc = []
    inf = []
    csim = []
    whichline = []
    num_phrase = []
    
    t0 = time()
    for line_idx in tqdm.tqdm(np.random.permutation(range(5000)[:1000])):
        
        line = linecache.getline(dfile, line_idx+1)
        sentence = BracketedSentence(line)
        if sentence.ntok<10:
            continue
        # orig = d[0]
        orig = sentence.words
        ntok = sentence.ntok
            
        phrase_idx = sentence.phrases(order=order, strict=True)
        if num_phrases is not None:
            if len(phrase_idx)<num_phrases:
                continue
        else:
            if len(phrase_idx)<1:
                continue
        if swap_type == 'within':
            perm_idx = [np.random.permutation(p) for p in phrase_idx]
        if phrase_type == 'imitation': # generate fake phrases of the same statistics
            pidx = []
            allidx = list(range(ntok))
            for p in phrase_idx:
                i = np.random.randint(0,np.max([1,len(allidx)-len(p)]))
                pidx.append(np.array(allidx[i:i+len(p)]))
                allidx = np.setdiff1d(allidx, allidx[i:i+len(p)])
            phrase_idx = pidx
            if swap_type == 'within':
                perm_idx = [np.random.permutation(p) for p in phrase_idx]
                assert len(np.unique(np.concatenate(perm_idx)))==sum([len(p) for p in perm_idx])
         
        orig_idx = np.array(range(ntok))
        if swap_type == 'within':
            swap_idx = np.array(range(ntok))
            for p in np.random.permutation(len(perm_idx))[:num_phrases]:
                swap_idx[phrase_idx[p]] = perm_idx[p]
        elif swap_type == 'among':
            boundaries = np.sort(np.concatenate([p[[0,-1]]+[0,1] for p in phrase_idx]))
            units = np.array([p for p in np.split(orig_idx, boundaries) if len(p)>0])
            is_phrs = np.array([np.all(np.isin(p,np.concatenate(phrase_idx))) for p in units])
            if num_phrases is not None:
                fixed = np.random.choice(np.nonzero(is_phrs)[0], sum(is_phrs)-num_phrases, replace=False)
                is_phrs[fixed]=False
            units[is_phrs] = np.random.permutation(units[is_phrs])
            swap_idx = np.concatenate(units.tolist())
            assert np.all(np.isin(np.unique(swap_idx), orig_idx))
            
        swapped = [orig[i] for i in swap_idx]    
        num_swaps.append(int(sum(orig_idx!=swap_idx)/2))
        
        # real
        orig_vecs = extract_tensor(orig, indices=orig_idx)
        swap_vecs = extract_tensor(swapped, indices=swap_idx)
        
        orig_centred = orig_vecs-orig_vecs.mean(axis=2, keepdims=True)
        swap_centred = swap_vecs-swap_vecs.mean(axis=2, keepdims=True)
        normalizer = (la.norm(orig_centred,2,1,keepdims=True)*la.norm(swap_centred,2,1,keepdims=True))
        csim.append(np.sum((orig_centred*swap_centred)/normalizer,1))
        
        frob.append(la.norm(orig_vecs-swap_vecs,'fro',axis=(1,2)))
        nuc.append(la.norm(orig_vecs-swap_vecs,'nuc',axis=(1,2)))
        inf.append(la.norm(orig_vecs-swap_vecs, np.inf,axis=(1,2)))
        
        whichline.append(line_idx)
        num_phrase.append(len(phrase_idx))
        
        # print('Done with line %d in %.3f seconds'%(i,time()-t0))
    
    fold = 'shuffle_test/'
    if swap_type == 'among':
        fold += 'ngrams/'
    
    if not os.path.isdir(SAVE_DIR+fold):
        os.makedirs(SAVE_DIR+fold)
    
    pref = '%s_%dorder'%(phrase_type, order)
    if num_phrases is not None:
        pref += '_%dphrases'%(num_phrases)
    np.save(open(SAVE_DIR+fold+pref+'_cosines.npy','wb'),np.concatenate(csim,axis=1))
    np.save(open(SAVE_DIR+fold+pref+'_frob.npy','wb'),np.stack(frob))
    np.save(open(SAVE_DIR+fold+pref+'_nuc.npy','wb'),np.stack(nuc))
    np.save(open(SAVE_DIR+fold+pref+'_inf.npy','wb'),np.stack(inf))
    np.save(open(SAVE_DIR+fold+pref+'_num_swap.npy','wb'),num_swaps)
    np.save(open(SAVE_DIR+fold+pref+'_line_id.npy','wb'),whichline)
    np.save(open(SAVE_DIR+fold+pref+'_num_phrase.npy','wb'), num_phrase)

print('Done!')




