SAVE_DIR = '/home/matteo/Documents/github/bertembeddings/data/'
CODE_DIR = '/home/matteo/Documents/github/bertembeddings/'

import sys
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

order = 1
num_phrases = None
phrase_type1 = 'real'
phrase_type2 = 'imitation'



real_cos = np.load(SAVE_DIR+'shuffle_test/%s_cosines.npy'%phrase_type1)
real_frob = np.load(SAVE_DIR+'shuffle_test/%s_frob.npy'%phrase_type1)
real_nuc = np.load(SAVE_DIR+'shuffle_test/%s_nuc.npy'%phrase_type1)
real_inf = np.load(SAVE_DIR+'shuffle_test/%s_inf.npy'%phrase_type1)
real_num_swap = np.load(SAVE_DIR+'shuffle_test/%s_num_swap.npy'%phrase_type1)
real_lines = np.load(SAVE_DIR+'shuffle_test/%s_line_id.npy'%phrase_type1)
real_num_phrase = np.load(SAVE_DIR+'shuffle_test/%s_num_phrase.npy'%phrase_type1)

fake_cos = np.load(SAVE_DIR+'shuffle_test/%s_cosines.npy'%phrase_type2)
fake_frob = np.load(SAVE_DIR+'shuffle_test/%s_frob.npy'%phrase_type2)
fake_nuc = np.load(SAVE_DIR+'shuffle_test/%s_nuc.npy'%phrase_type2)
fake_inf = np.load(SAVE_DIR+'shuffle_test/%s_inf.npy'%phrase_type2)
fake_num_swap = np.load(SAVE_DIR+'shuffle_test/%s_num_swap.npy'%phrase_type2)
fake_lines = np.load(SAVE_DIR+'shuffle_test/%s_line_id.npy'%phrase_type2)
fake_num_phrase = np.load(SAVE_DIR+'shuffle_test/%s_num_phrase.npy'%phrase_type2)