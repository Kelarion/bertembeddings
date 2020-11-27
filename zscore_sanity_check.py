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

def ngram_shuffling(idx, n):
    """ 
    idx is a n_token length sequence of indices, n is the size of a chunk 
    ensure that n<len(idx)
    """
    if len(idx)<n:
        raise ValueError
    n_pad = int(n-np.mod(len(idx),n))
    # padded = np.insert(swap_idx.astype(float), 
    #                    ntok-n_pad-1, 
    #                    np.ones(n_pad)*np.nan)
    padded = np.append(idx.astype(float), np.ones(n_pad)*np.nan)
    while 1:
        shuf_idx = np.random.permutation(padded.reshape((-1,n))).flatten()
        shuf_idx = shuf_idx[~np.isnan(shuf_idx)].astype(int)
        if np.any(shuf_idx != idx):
            break
    # shuf_idx = np.random.permutation(padded.reshape((-1,n))).flatten()
    # shuf_idx = shuf_idx[~np.isnan(shuf_idx)].astype(int)
    return shuf_idx

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
print('Computing mean and variance ...\n')
# foo = []
all_vecs = []
for line_idx in tqdm.tqdm(np.random.choice(5000,500,replace=False)):
    
    line = linecache.getline(dfile, line_idx+1)
    sentence = BracketedSentence(line)
    orig = sentence.words
    ntok = sentence.ntok
    if ntok < 10:
        continue
    # orig = d[0]
        
    orig_idx = np.array(range(ntok))
    
    n = np.random.choice(range(1,6))
    # n=1
    swap_idx = ngram_shuffling(np.arange(ntok), n)
    swapped = [orig[i] for i in swap_idx]
    
    # assert(swapped == line[1+phrase_type][swap_type][2])
    assert([swapped[i] for i in np.argsort(swap_idx)] == orig)
    
    # real
    orig_vecs = extract_tensor(orig)
    swap_vecs = extract_tensor(swapped, indices=np.argsort(swap_idx))
    
    all_vecs.append(np.append(orig_vecs, swap_vecs, -1))
m = np.concatenate(all_vecs,-1).mean(-1,keepdims=True)
s = np.concatenate(all_vecs,-1).std(-1,keepdims=True)

#%%
rank = 4
noise = 0.2
line_idx = 2

real_data = False
# real_data = True

line = BracketedSentence(linecache.getline(dfile, line_idx+1))

swap_idx = ngram_shuffling(np.arange(line.ntok),1)

vecs = extract_tensor(line.words)

if real_data:
    scale = vecs.var(axis=(1,2), keepdims=True)
else:
    scale = (1-np.exp(-0.2*np.arange(13))[:,None,None]+0.5)
    scale[-1] = 0.1
scale = np.linspace(0.5,10,20)[:,None,None]
# swap_vecs = extract_tensor([line.words[i] for i in swap_idx], indices=np.argsort(swap_idx))


# frobs = []
# for _ in range(100):

z0 = vecs[0,...] # embedding layer
dz = np.random.randn(768, rank)@np.random.randn(rank,20)*noise
z0_swp = z0+dz
# z0_swp = swap_vecs[0,...]

if real_data:
    fake_orig = vecs
    fake_swapped = swap_vecs
else:
    fake_orig = z0[None,:,:]*scale
    fake_swapped = z0_swp[None,:,:]*scale    

m = np.append(fake_orig,fake_swapped,axis=-1).mean(-1, keepdims=True)
s = np.append(fake_orig,fake_swapped,axis=-1).std(-1, keepdims=True)

nrm = la.norm(np.append(fake_orig,fake_swapped,axis=-1),axis=1,keepdims=True).mean(-1,keepdims=True)
std = np.append(fake_orig,fake_swapped,axis=-1).std((1,2), keepdims=True)

diff_naive = fake_orig - fake_swapped
diff_zscored = (fake_orig)/s - (fake_swapped-m)/s
diff_rescaled = (fake_orig)/nrm - (fake_swapped)/nrm
diff_stand = (fake_orig)/std - (fake_swapped)/std

frob_naive = la.norm(diff_naive, 'fro', axis=(1,2))/np.sqrt(np.prod(vecs.shape[1:]))
frob_zscored = la.norm(diff_zscored, 'fro', axis=(1,2))/np.sqrt(np.prod(vecs.shape[1:]))
frob_rescaled = la.norm(diff_rescaled, 'fro', axis=(1,2))/np.sqrt(np.prod(vecs.shape[1:]))
frob_stand = la.norm(diff_stand, 'fro', axis=(1,2))/np.sqrt(np.prod(vecs.shape[1:]))

# frobs.append([frob_naive, frob_zscored])
plt.plot(frob_naive)
plt.plot(frob_zscored)
plt.plot(scale.squeeze(),'k--')
plt.ylim([0,plt.ylim()[1]])


#%%

plt.plot(scale.squeeze(),frob_naive,linewidth=2)
plt.plot(scale.squeeze(),frob_zscored,linewidth=2)
plt.plot(scale.squeeze(),frob_rescaled,linewidth=2)
plt.plot(scale.squeeze(),frob_stand,linewidth=2)





