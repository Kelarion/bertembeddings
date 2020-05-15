SAVE_DIR = '/home/matteo/Documents/github/bertembeddings/data/'
CODE_DIR = '/home/matteo/Documents/github/'

import sys
sys.path.append(CODE_DIR+'repler/src/')

import torch

from transformers import BertTokenizer, BertModel, BertConfig
import pickle as pkl

import numpy as np

import os
import random
from time import time

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

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
random_model = False

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

if random_model:
    model = BertModel(BertConfig(output_hidden_states=True))
    base_directory = 'vectors/permuted_depth/bert_untrained/'
else:
    model = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)
    base_directory = 'extracted/bert-base-cased/'

print('Beginning extraction ... ')
with open(SAVE_DIR+'phrase_boundary_tree_dist.pkl', 'rb+') as dfile:
    for line_idx, line in enumerate(pkl.load(dfile)):
        t0 = time()
        directory = base_directory + str(line_idx) + '/'
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        orig_sent = line[0]
        swapped_sent = line[1]
        swapped_idx = line[2]
        depth_diff = line[3]
        swapped_idx_in_same_phrase = line[4]

        original_indices = list(range(len(orig_sent)))

        swapped_indices = list(range(len(orig_sent)))
        swapped_indices[swapped_idx[0]] = swapped_idx[1]
        swapped_indices[swapped_idx[1]] = swapped_idx[0]

        # original sentence
        original_vectors = extract_tensor(orig_sent, original_indices)

        # swapped sentence
        swapped_vectors = extract_tensor(swapped_sent, swapped_indices)

        pkl.dump(original_vectors, open(directory + 'original_vectors.pkl', 'wb+'))
        pkl.dump(swapped_vectors, open(directory + 'swapped_vectors.pkl', 'wb+'))
        
        print('Done with line %d, after %.3f seconds'%(line_idx, time()-t0))












