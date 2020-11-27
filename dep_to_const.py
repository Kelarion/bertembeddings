
import numpy as np
import pandas

import socket
if socket.gethostname() == 'kelarion':
    CODE_DIR = 'C:/Users/mmall/Documents/github/bertembeddings/'
    SAVE_DIR = 'C:/Users/mmall/Documents/github/bertembeddings/data/'
    LOAD_DIR = 'C:/Users/mmall/Documents/github/bertembeddings/data/extracted/'
else:    
    CODE_DIR = '/home/malleman/bert_code/'
    LOAD_DIR = '/om3/group/chung/cca/vectors/swapped/bert-base-cased/'
    SAVE_DIR = '/om2/user/malleman/bert/'

import sys, os
# sys.path.append(CODE_DIR+'repler/src/')

sys.path.append(CODE_DIR)
from brackets2trees import BracketedSentence

#%%
# chunk the job into 40 subjobs doing 954 lines each

allargs = sys.argv

arglist = allargs[1:]

which_job = int(allargs[1])
start = which_job*954
if which_job == 39:
    stop = 38186
else:
    stop = start+954

svfolder = '%d-%d'%(start,stop)
if not os.path.isdir(SAVE_DIR+svfolder):
    os.makedirs(SAVE_DIR+svfolder)

#%%

dep = list(open(SAVE_DIR+'/dependency_train_bracketed.txt', 'r'))

# tok_num = dep[0].values
# line_num = np.cumsum(np.diff(tok_num, prepend=0)<0)
# tokens = dep[1].values
# sent_length = np.unique(line_num, return_counts=True)[1]
# sent_length = [BracketedSentence(d,True).ntok for d in dep]
sent_length = []
tokens = []
for d in dep:
    bs = BracketedSentence(d,True)
    sent_length.append(bs.ntok)
    tokens.append(bs.words)

#%%
const = open(SAVE_DIR+'/train_bracketed.txt','r')

unmatched_lines = list(range(len(dep)))
idx_in_train = np.zeros(len(unmatched_lines))*np.nan
for line_idx, line in enumerate(const):
    if line_idx<start:
        continue
    if line_idx>stop:
        break
    
    words = BracketedSentence(line).words
    
    # randomly sample 
    possible_lines = [i for i in unmatched_lines if sent_length[i]==len(words)]
    # same_line = [np.all([tokens[line_num==j][i] == words[i] for i in range(len(words))]) for j in possible_lines]
    found = False
    for i in np.random.permutation(possible_lines):
        # sameline = np.all([tokens[i][j] == words[j] for j in range(len(words))])
        if tokens[i] == words:
            unmatched_lines.remove(i)
            idx_in_train[i] = line_idx
            found = True
            break
    if found:
        print('Matched line %d!'%line_idx)
    else:
        print('No match for line %d :('%line_idx)
        print('... it was %d words, btw'%len(words))    

np.save(SAVE_DIR+svfolder+'/dep_line_idx.npy',idx_in_train)


