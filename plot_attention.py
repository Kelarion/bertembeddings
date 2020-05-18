SAVE_DIR = '/home/matteo/Documents/github/bertembeddings/data/'
CODE_DIR = '/home/matteo/Documents/github/bertembeddings/'

import sys
sys.path.append(CODE_DIR+'repler/src/')

import torch

from transformers import BertTokenizer, BertModel, BertConfig
import pickle as pkl
import numpy as np


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
        all_output = model(input_ids)
        
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

    return np.stack(layer_vectors), all_output

#%%

model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


with open(SAVE_DIR+'permuted_data.pkl', 'rb') as dfile:
    dist = pkl.load(dfile)

#%%
win = 2 # how many words to take around the phrase
gram = 4

real_grams = []
real_grams_swp = []

fake_grams = []

for d in np.random.permutation(dist)[:1000]:
    orig = d[0]
    
    # real
    inds = d[1]
    phr1 = range(inds[0][0]-win,inds[0][1]+win+1)
    phr1_len = int(np.diff(inds[0])+1)
    phr2 = range(inds[1][0]-win,inds[1][1]+win+1)
    phr2_len = int(np.diff(inds[1])+1)
    ba, wa = extract_tensor(orig)
    
    if (phr1[0]>0) and (phr1[-1]<len(orig)) and (phr1_len==gram):
        mat1 = np.zeros((12,12,phr1_len+2*win,phr1_len+2*win))
        for l in range(12):
            attn = wa[3][l][0,:,1:-1,1:-1]
            mat1[l,:,:,:] = attn[:,phr1,:][:,:,phr1]
        real_grams.append(mat1)
        
        # swap first and last words
        swp_idx = list(range(len(orig)))
        swp_idx[phr1[win]] = phr1[-win-1]
        swp_idx[phr1[-win-1]] = phr1[win]
        
        ba_, wa_ = extract_tensor([orig[j] for j in swp_idx])

        mat2 = np.zeros((12,12,phr1_len+2*win,phr1_len+2*win))
        for l in range(12):
            attn = wa_[3][l][0,:,1:-1,1:-1]
            attn = attn[:,swp_idx,:][:,:,swp_idx]
            mat2[l,:,:,:] = wa_[3][l][0,:,phr1,:][:,:,phr1]
        real_grams_swp.append(mat2)
    
    if (phr2[0]>0) and (phr2[-1]<len(orig)) and (phr2_len==gram):
        mat1 = np.zeros((12,12,phr2_len+2*win,phr2_len+2*win))
        for l in range(12):
            attn = wa[3][l][0,:,1:-1,1:-1]
            mat1[l,:,:,:] = attn[:,phr2,:][:,:,phr2]
        real_grams.append(mat1)
        
        # swap first and last words
        swp_idx = list(range(len(orig)))
        swp_idx[phr2[win]] = phr2[-win-1]
        swp_idx[phr2[-win-1]] = phr2[win]
        
        ba_, wa_ = extract_tensor([orig[j] for j in swp_idx])

        mat2 = np.zeros((12,12,phr2_len+2*win,phr2_len+2*win))
        for l in range(12):
            attn = wa_[3][l][0,:,1:-1,1:-1]
            attn = attn[:,swp_idx,:][:,:,swp_idx]
            mat2[l,:,:,:] = wa_[3][l][0,:,phr2,:][:,:,phr2]
        real_grams_swp.append(mat2)
    
    # fake
    inds = d[2]
    phr1 = range(inds[0][0]-win,inds[0][1]+win+1)
    phr1_len = int(np.diff(inds[0])+1)
    phr2 = range(inds[1][0]-win,inds[1][1]+win+1)
    phr2_len = int(np.diff(inds[1])+1)
    ba, wa = extract_tensor(orig)
    
    if (phr1[0]>0) and (phr1[-1]<len(orig)) and (phr1_len==gram):
        mat1 = np.zeros((12,12,phr1_len+2*win,phr1_len+2*win))
        for l in range(12):
            mat1[l,:,:,:] = wa[3][l][0,:,phr1,:][:,:,phr1]
        fake_grams.append(mat1)
    
    if (phr2[0]>0) and (phr2[-1]<len(orig)) and (phr2_len==gram):
        mat1 = np.zeros((12,12,phr2_len+2*win,phr2_len+2*win))
        for l in range(12):
            mat1[l,:,:,:] = wa[3][l][0,:,phr2,:][:,:,phr2]
        fake_grams.append(mat1)
    
    print('Got %d real and %d fake %d-grams'%(len(real_grams),len(fake_grams),gram))

# trigs = np.stack(real_)
# phrigs = np.trigramsstack(fake_trigrams)
real_original = np.stack(real_grams)
real_swapped = np.stack(real_grams_swp)
fake_original = np.stack(fake_grams)

#%%
T = real_original.mean(0)
T = real_swapped.mean(0)
# T = fake_original.mean(0)

plt.figure()
for l in range(12):
    for h in range(12):
        n = 12*l + h + 1
        plt.subplot(12,12,n)
        plt.imshow(T[l,h,:,:])
        plt.xticks([])
        plt.yticks([])
        # plt.clim([0,1])

