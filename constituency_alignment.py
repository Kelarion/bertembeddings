SAVE_DIR = '/home/matteo/Documents/github/bertembeddings/data/'
CODE_DIR = '/home/matteo/Documents/github/bertembeddings/'

import sys
sys.path.append(CODE_DIR)
from brackets2trees import BracketedSentence

import torch

from transformers import BertTokenizer, BertModel, BertConfig
import pickle as pkl
import numpy as np
import linecache
from time import time
import matplotlib.pyplot as plt

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
        attn_weight = model(input_ids)[3]
        
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
            
            if layer>0:
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
        if layer>0:
            attn_matrices = np.stack(list_of_attn).transpose(1,0,2)
            layer_attns.append(attn_matrices)

    return np.stack(layer_vectors), np.stack(layer_attns)

#%%

model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


dfile = SAVE_DIR+'train_bracketed.txt'

# with open(SAVE_DIR+'permuted_data.pkl', 'rb') as dfile:
#     dist = pkl.load(dfile)

#%%
# avg_type = 'tril' # only consider precedent words
# avg_type = 'triu' # only consider subsequent words
avg_type = 'full' # only consider subsequent words
include_diag = True

order = 2
phrase_type = 'real' 
# phrase_type = 'scramble' 
# phrase_type = 'unreal' 
phrase_type = 'blocks' 
# phrase_type = 'all' 
phrase_window = None
all_window = 4

sum_align = []
sum_attn = []
num_in_phrase = []
num_pairs = [] # number of pairs we compare, to take averages
dt_inphrase = [] # distance of words to other words in the same phrase
t0 = time()
for line_idx in np.random.permutation(range(1000)):
    
    line = linecache.getline(dfile, line_idx+1)
    sentence = BracketedSentence(line)
    if sentence.ntok<10:
        continue
    # orig = d[0]
    orig = sentence.words
    ntok = sentence.ntok
    
    # whether to use true phrases
    if phrase_type == 'real':
        const = np.array([[sentence.is_relative(i,j, order=order) \
                           for i in range(ntok)] \
                              for j in range(ntok)])
    elif phrase_type == 'unreal':
        const = np.array([[(not sentence.is_relative(i,j, order=order)) \
                               for i in range(ntok)] \
                                  for j in range(ntok)])
    elif phrase_type == 'scramble':
        const = np.array(
            [[sentence.is_relative(i,np.min([j+np.random.randint(0,3), ntok-1]), order=order) \
                           for i in range(ntok)] \
                              for j in range(ntok)])
        # const += np.tril(const).T + np.triu(const).T
    elif phrase_type == 'all':
        const = np.ones((ntok,ntok))
    
    # whether to look only at nearby or far words
    if phrase_window is not None:
        if np.sign(phrase_window)>0:
            phr_win = np.array([[np.abs(i-j)<phrase_window \
                               for i in range(ntok)] \
                                  for j in range(ntok)])
        else:
            phr_win = np.array([[np.abs(i-j)>-phrase_window \
                               for i in range(ntok)] \
                                  for j in range(ntok)])
    else:
        phr_win = np.ones((ntok,ntok))
    
    if all_window is not None:
        if np.sign(all_window)>0:
            win = np.array([[np.abs(i-j)<all_window \
                               for i in range(ntok)] \
                                  for j in range(ntok)])
        else:
            win = np.array([[np.abs(i-j)>-all_window \
                               for i in range(ntok)] \
                                  for j in range(ntok)])
    else:
        win = np.ones((ntok,ntok))
    
    if include_diag:
        const[np.arange(ntok),np.arange(ntok)]=True # keep diagonal
        win[np.arange(ntok),np.arange(ntok)]=True # keep diagonal
    else:
        const[np.arange(ntok),np.arange(ntok)]=False 
        win[np.arange(ntok),np.arange(ntok)]=False # keep diagonal
    
    
    dt = [np.nanmax([np.abs(i-j) if sentence.is_relative(i,j,order=order) else np.nan for i in range(j+1)]) \
          for j in range(ntok)]
    dt_inphrase.append(np.mean(np.array(dt)[np.array(dt)!=0]))
    
    # real
    _, attn = extract_tensor(orig)
    # for l in range(12):
    if avg_type == 'tril':
        d = [[np.sum(phr_win*np.tril(const)*attn[i,j,:,:]) \
             for i in range(12)] \
             for j in range(12)]
        a = [[np.sum(win*np.tril(attn[i,j,:,:])) \
             for i in range(12)] \
             for j in range(12)]
        num_in_phrase.append(np.sum(np.tril(phr_win*const)))
        num_pairs.append(np.sum(np.tril(win)))
    elif avg_type == 'triu':
        d = [[np.sum(phr_win*np.triu(const)*attn[i,j,:,:]) \
             for i in range(12)] \
             for j in range(12)]
        a = [[np.sum(win*np.triu(attn[i,j,:,:])) \
             for i in range(12)] \
             for j in range(12)]
        num_in_phrase.append(np.sum(np.triu(phr_win*const)))
        num_pairs.append(np.sum(np.triu(win)))
    elif avg_type == 'full':
        d = [[np.sum(phr_win*const*attn[i,j,:,:]) \
             for i in range(12)] \
             for j in range(12)]
        a = [[np.sum(win*attn[i,j,:,:]) \
             for i in range(12)] \
             for j in range(12)]
        num_in_phrase.append(np.sum(phr_win*const))
        num_pairs.append(np.sum(win))
    sum_align.append(np.array(d))
    sum_attn.append(np.array(a))
    
    print('Done with line %d in %.3f seconds'%(line_idx,time()-t0))

# print(np.sum(d)/np.sum(a))
alignment = np.stack(sum_align).sum(0)/np.stack(sum_attn).sum(0)

#%%

mean_attn_inphrase = (np.stack(sum_align)/np.array(num_in_phrase)[:,None,None])
mean_attn_local = (np.stack(sum_attn)/np.array(num_pairs)[:,None,None])

for l in range(12):
    for h in range(12):
        n_sub = 12*l + h + 1
        plt.subplot(12,12,n_sub)
        plt.hist(mean_attn_inphrase[:,l,h], density=True, alpha=0.7)
        plt.hist(mean_attn_local[:,l,h], density=True, alpha=0.7)
        plt.xticks([])
        plt.yticks([])
        if h==0:
            plt.ylabel('Layer %d'%l)
        if l==11:
            plt.xlabel('Head %d'%h)


