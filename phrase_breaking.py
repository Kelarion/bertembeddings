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

dist = pkl.load(open('C:/Users/mmall/Documents/github/bertembeddings/data/phrase_swaps/permuted_data_np_vp.pkl','rb'))

# jon_folder = 'C:/Users/mmall/Documents/github/bertembeddings/data/jonathans/'
# with open(jon_folder+'phrase/permuted_data.pkl', 'rb') as f:
#     dist = pkl.load(f)
# lines = os.listdir(jon_folder+'/ngram/bert-base-cased/')

#%%
order = 1
# num_phrases = None
num_phrases = None

# swap_type = 'within'
swap_type = 'among'

# print('Computing mean and variance ...')
all_vecs = []
for line in tqdm.tqdm(np.random.permutation(dist)[:500]):
    
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
        if len(phrase_idx)<2 or (np.sum(np.isin(range(ntok),np.concatenate(phrase_idx))) > ntok/2):
            continue
    
    # if phrase_type == 'imitation': # generate fake phrases of the same statistics
    pidx = []
    allidx = np.setdiff1d(range(ntok),[ph[0] for ph in phrase_idx])
    for phrs in phrase_idx:
        i_max = ntok-len(phrs)
        i = np.random.choice(allidx[allidx<i_max])
        pidx.append(np.arange(i,i+len(phrs)))
        allidx = np.setdiff1d(allidx, np.arange(i,i+len(phrs)))
    phrase_idx = pidx

    orig_idx = np.array(range(ntok))
    
    boundaries = np.sort(np.concatenate([p[[0,-1]]+[0,1] for p in phrase_idx]))
    units = np.array([p for p in np.split(orig_idx, boundaries) if len(p)>0])
    is_phrs = np.array([np.all(np.isin(p,np.concatenate(phrase_idx))) for p in units])
    if num_phrases is not None:
        fixed = np.random.choice(np.nonzero(is_phrs)[0], sum(is_phrs)-num_phrases, replace=False)
        is_phrs[fixed]=False
    pswp = np.sort(np.random.choice(np.where(is_phrs)[0],2,replace=False))
    units[pswp] = np.flip(units[pswp])
    # units[is_phrs] = np.random.permutation(units[is_phrs])
    swap_idx = np.concatenate(units.tolist())
    assert np.all(np.isin(np.unique(swap_idx), orig_idx))
    
    swapped = [orig[i] for i in swap_idx]    
    
    # real
    orig_vecs = extract_tensor(orig, indices=orig_idx)
    swap_vecs = extract_tensor(swapped, indices=swap_idx)
    
    all_vecs.append(np.append(orig_vecs, swap_vecs, -1))
m = np.concatenate(all_vecs,-1).mean(-1,keepdims=True)
s = np.concatenate(all_vecs,-1).std(-1,keepdims=True)


num_swaps = [] # track the number of words swapped in each sentence
frob = []
nuc = []
inf = []
csim = []
whichline = []
whichcond = []
num_phrase = []

t0 = time()
for line_idx in tqdm.tqdm(np.random.permutation(range(5000))[:1000]):
    
    line = linecache.getline(dfile, line_idx+1)
    sentence = BracketedSentence(line)
    if sentence.ntok<10:
        continue
    # orig = d[0]
    orig = sentence.words
    ntok = sentence.ntok
    
    orig_idx = np.array(range(ntok))
        
    phrase_idx = sentence.phrases(order=order, strict=True)
    if len(phrase_idx)<2 or (np.sum(np.isin(range(ntok),np.concatenate(phrase_idx))) > ntok/2):
            continue
    
    # choose real phrases
    boundaries = np.sort(np.concatenate([p[[0,-1]]+[0,1] for p in phrase_idx]))
    units = np.array([p for p in np.split(orig_idx, boundaries) if len(p)>0])
    is_phrs = np.array([np.all(np.isin(p,np.concatenate(phrase_idx))) for p in units])
    pswp = np.sort(np.random.choice(np.where(is_phrs)[0],2,replace=False))
    units[pswp] = np.flip(units[pswp])
    # units[is_phrs] = np.random.permutation(units[is_phrs])
    
    # choose fake phrases
    
    for pbin, phrase_type in enumerate(['real','imitation']):
        
        if phrase_type == 'real':
            swap_idx = np.concatenate(units.tolist())            
        elif phrase_type == 'imitation':
            fake_phrase_idx = []
            allidx = np.setdiff1d(range(ntok),[ph[0] for ph in phrase_idx])
            for phrs in units[pswp]:
                i_max = ntok-len(phrs)
                i = np.random.choice(allidx[allidx<i_max])
                fake_phrase_idx.append(np.arange(i,i+len(phrs)))
                allidx = np.setdiff1d(allidx, np.arange(i,i+len(phrs)))
            boundaries = np.sort(np.concatenate([p[[0,-1]]+[0,1] for p in fake_phrase_idx]))
            units = np.array([p for p in np.split(orig_idx, boundaries) if len(p)>0])
            is_phrs = np.array([np.all(np.isin(p,np.concatenate(fake_phrase_idx))) for p in units])
            pswp = np.sort(np.random.choice(np.where(is_phrs)[0],2,replace=False))
            units[pswp] = np.flip(units[pswp])
            swap_idx = np.concatenate(units.tolist())    
        
        swapped = [orig[i] for i in swap_idx]    
        num_swaps.append(int(sum(orig_idx!=swap_idx)/2))
        
        # real
        orig_vecs = extract_tensor(orig, indices=orig_idx)
        swap_vecs = extract_tensor(swapped, indices=swap_idx)
        
        orig_vecs_zscore = (orig_vecs-m)/s
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
        num_phrase.append(len(phrase_idx))
        whichcond.append(pbin)
        
        # print('Done with line %d in %.3f seconds'%(i,time()-t0))


fold = 'phrase_swaps/'
if swap_type == 'within':
    fold += 'within/'

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
np.save(open(SAVE_DIR+fold+pref+'_num_swap.npy','wb'),num_swaps)
np.save(open(SAVE_DIR+fold+pref+'_line_id.npy','wb'),whichline)
np.save(open(SAVE_DIR+fold+pref+'_num_phrase.npy','wb'), num_phrase)
np.save(open(SAVE_DIR+fold+pref+'_phrase_type.npy','wb'), whichcond)

print('Done!')



