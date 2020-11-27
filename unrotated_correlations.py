SAVE_DIR = 'C:/Users/mmall/Documents/github/bertembeddings/data/'
CODE_DIR = 'C:/Users/mmall/Documents/github/bertembeddings/'

import sys, os
sys.path.append(CODE_DIR)
from brackets2trees import BracketedSentence

import torch
import tqdm

from transformers import BertTokenizer, BertModel, BertConfig, AutoConfig, AutoModel
import pickle as pkl
import numpy as np
import scipy.linalg as la
from sklearn.decomposition import PCA
from pwcca import compute_pwcca
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

def correlate(X, Y):
    """
    Pearson correlation of each dimension of X and Y, assuming that the last axis is samples
    """
    # N = X.shape[-1]

    cov = np.sum((X-X.mean(-1,keepdims=True))*(Y-Y.mean(-1,keepdims=True)),-1)
    varx = np.sum((X-X.mean(-1,keepdims=True))**2,-1)
    vary = np.sum((Y-Y.mean(-1,keepdims=True))**2,-1)
    
    return cov/np.sqrt(varx*vary)
    # cov = np.cov(vecs[0,0,...],vecs[1,0,...])
    # xx = cov[:768,:768]
    # yy = cov[768:,768:]
    # xy = cov[:768,768:]
    
    # return np.diag(xy)/np.sqrt(np.diag(xx)*np.diag(yy))

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
max_num = 300
these_bounds = [0,1,2,3,4]
n_pca = 100

frob = []
nuc = []
inf = []
csim = []
avgdist = []
whichline = []
whichcond = []
whichswap = []
norms = []
# mean = np.zeros((13,768))

num_cond = np.zeros(len(these_bounds))
corrs = [[] for _ in these_bounds]
cca = [[] for _ in these_bounds]
# corrs = np.zeros((max_num, len(these_bounds)))
t0 = time()
pbar = tqdm.tqdm(total=max_num*len(these_bounds), desc='Tree dist 0/%d'%len(these_bounds))
for dt in these_bounds:
    pbar.desc = 'Tree dist %d/%d'%(dt,len(these_bounds))
    vecs = []
    whichline = []
    whichswap = []
    for line_idx in np.random.permutation(range(5000)):
        
        line = linecache.getline(dfile, line_idx+1)
        sentence = BracketedSentence(line)
        if sentence.ntok<10:
            continue
        # orig = d[0]
        orig = sentence.words
        ntok = sentence.ntok
        
        crossings = np.diff(np.abs(sentence.brackets).cumsum()[sentence.term2brak])
        these_pairs = np.isin(crossings,dt)
        if not np.any(these_pairs):
            continue
        
        orig_idx = np.array(range(ntok))
        
        i = np.random.choice(np.where(these_pairs)[0])
        c = crossings[i]
        # for i,c in zip(np.where(these_pairs)[0],crossings[these_pairs]):
        if (num_cond[c] >= max_num):
            continue
        num_cond[c] += 1
        
        swap_idx = np.array(range(ntok))
        swap_idx[i+1] = i
        swap_idx[i] = i+1
        
        swapped = [orig[i] for i in swap_idx]
        
        # real
        orig_vecs = extract_tensor(orig, indices=orig_idx)
        swap_vecs = extract_tensor(swapped, indices=swap_idx)
        
        vecs.append([orig_vecs,swap_vecs])
        # mean += np.append(orig_centred, swap_centred, -1).sum(-1)
        
        whichline.append(line_idx)
        # whichcond.append(c)
        whichswap.append(np.repeat(len(whichline), ntok))
        
        pbar.update(1)
         
        if (num_cond[c] >= max_num):
            break
    
    vecs = np.concatenate(vecs,axis=-1)
    # together = np.append(vecs[0,...], vecs[1,...], axis=-1)
    
    # a hack to make sure the matrices aren't rank decifient
    # n_pca = np.min([np.linalg.matrix_rank(OG[l,:,:]), pca_comp])
    # pca = PCA(n_components=n_pca)
    # print('Using %d components'%n_pca)
    corrs[dt].append(correlate(vecs[0,...],vecs[1,...]))
    cca_corr = []
    for l in range(13):
        # pca.fit(together[l,...].T)
        # M1 = pca.transform(vecs[0,l,...].T).T
        # M2 = pca.transform(vecs[1,l,...].T).T
        cca_corr.append(compute_pwcca(vecs[0,l,...],vecs[1,l,...])[0])
        # ccc.append(correlate(M1,M2))
    cca[dt].append(cca_corr)

#%%

fold = 'bracket_crossings/full_features/'
if random_model:
    fold += 'random_model/'

if not os.path.isdir(SAVE_DIR+fold):
    os.makedirs(SAVE_DIR+fold)

# pref = '%s_%dorder'%(phrase_type, order)

np.save(open(SAVE_DIR+fold+'_cosines.npy','wb'),np.concatenate(csim,axis=1))
np.save(open(SAVE_DIR+fold+'_frob.npy','wb'),np.stack(frob))
np.save(open(SAVE_DIR+fold+'_nuc.npy','wb'),np.stack(nuc))
np.save(open(SAVE_DIR+fold+'_inf.npy','wb'),np.stack(inf))
np.save(open(SAVE_DIR+fold+'_line_id.npy','wb'),whichline)
np.save(open(SAVE_DIR+fold+'_num_crossings.npy','wb'), whichcond)
np.save(open(SAVE_DIR+fold+'_swap_id.npy','wb'), np.concatenate(whichswap))
np.save(open(SAVE_DIR+fold+'_average_norms.npy','wb'), np.concatenate(norms, -1))
# np.save(open(SAVE_DIR+fold+'_average_norms.npy','wb'), np.concatenate(norms, -1))
np.save(open(SAVE_DIR+fold+'_dist_avg.npy','wb'), np.stack(avgdist))

print('Done!')




