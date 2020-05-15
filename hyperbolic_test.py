SAVE_DIR = '/home/matteo/Documents/github/bertembeddings/'
CODE_DIR = '/home/matteo/Documents/github/'

import sys

from tqdm import tqdm, trange
from time import sleep
import pickle as pkl

import numpy as np
import pandas
import torch 
import torch.nn as nn
import torch.optim as optim

import scipy.special as spc
import scipy.linalg as la
import scipy.sparse as sprs
import matplotlib.pyplot as plt
from sklearn import svm, discriminant_analysis, manifold

# from torch.autograd import Function

# my code
sys.path.append(CODE_DIR+'repler/src/')
from hyperbole.hyperbolic_utils import Hyperboloid, PseudoPolar, TangentSpace
from hyperbole.dataset_utils import DenseDataset, SparseGraphDataset
from students import Feedforward 
from assistants import Indicator

sys.path.append(CODE_DIR+'bertembeddings/')
from brackets2trees import BracketedSentence

#%%
def poincare_dist(x,y):
    d = 2+np.sum((x-y)**2, -1)/((1-np.sum(x**2))*(1-np.sum(y**2)))
    return np.arccosh(1+d)

#%%
layer = -1
line_idx = 4

# dist = pkl.load(open(SAVE_DIR+'data/phrase_boundary_tree_dist.pkl','rb'))

# swap_sim = pkl.load(open(SAVE_DIR+'data/line%d_swapsimilarity.pkl'%line_idx,'rb'))
# weights = list(np.array(swap_sim[layer])[:,0])
# idx = np.array(list(np.array(swap_sim[layer])[:,1]))

# tree_dists = pkl.load(open(SAVE_DIR+'data/line%d_treedist.pkl'%line_idx,'rb'))
# weights = list((-np.array(tree_dists)[:,0]))
# idx = np.array(list(np.array(tree_dists)[:,1]))

# og = pkl.load(open(SAVE_DIR+'extracted/bert-base-cased/'+str(line_idx)+'/original_vectors.pkl','rb'))

# X = og[layer,:,:].T.dot(og[layer,:,:])/la.norm(og[layer,:,:], axis=0)**2
# X = -la.norm(og[-1,:,:,None]-og[-1,:,None,:],axis=0)
# seq = dist[line_idx][0]'

with open(SAVE_DIR+'data/train_bracketed.txt', 'r') as dfile:
    for i in range(line_idx):
        line = dfile.readline()

brak = BracketedSentence(line)

# weights = []
# idx = []
# for i in range(len(brak.terminals)):
#     for j in range(i+1,len(brak.terminals)):
#         weights.append(-brak.tree_dist(i,j))
#         idx.append((i,j))
# idx = np.array(idx)
# seq = brak.words

weights = []
idx = []
for i in range(len(brak.nodes)):
    for j in range(i+1,len(brak.nodes)):
        weights.append(-brak.tree_dist(i,j,False))
        idx.append((i,j))
idx = np.array(idx)
seq = brak.nodes

#%%
dim = 2 # dimension of poincare embedding

n_neg = 10
# n_neg = 0

bsz = 32
nepoch = 1000
eta = 0.3

burnin = 50
c_bi = 10

# D = DenseDataset(X, seq, bsz, n_neg=n_neg, padding=False)
# D = DenseDataset(np.diag(np.diagonal(X,1),1), seq, bsz, n_neg=n_neg, padding=False)
D = SparseGraphDataset(idx, weights, seq, bsz, n_neg=n_neg)
U = Hyperboloid(len(seq), dim+1, init_range=1e-3, padding=False)

train_loss = np.zeros(nepoch)
for epoch in range(nepoch):
    
    lr = eta
    if epoch <= burnin:
        lr /= c_bi
        
    running_loss = 0
    with tqdm(D, total=D.num_batches, desc='Epoch %d'%epoch, postfix=[dict(loss_=0)]) as looper:
        for i, (n,t) in enumerate(looper):
            
            U.zero_grad()
            
            logprob = -U.distances(n)
            # loss = nn.PoissonNLLLoss()(logprob, t)
            # loss = torch.abs(t.float() - U.distances(n)**2).sum()
            
            # logprob = -U.distances(n)
            loss = nn.CrossEntropyLoss()(logprob, t)
            loss.backward()
            
            dL = U.weight.grad.data # euclidean gradient
            assert torch.all(dL==dL), "Contains NaN"
            
            gradL = U.rgrad(U.weight.data, dL) # project onto tangent space
            gradL = U.proj(U.weight.data, gradL)
            
            u_ = U.expmap(U.weight, -lr*gradL) # exponential map
            assert torch.all(u_==u_), "Contains NaN"
            u_ = U.normalize(u_)
            U.weight.data.copy_(u_) # update
            
            running_loss += loss.item()
            
            looper.postfix[0]['loss_'] = np.round(running_loss/(i+1), 3)
            looper.update()
            
    # print('Epoch %d, loss=%.3f'%(epoch, running_loss/(i+1)))    
    train_loss[epoch] = running_loss/(i+1)

#%%
poinc = U.weight.data[:,1:]/(1+U.weight.data[:,:1])

plt.scatter(poinc[:,0],poinc[:,1])

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

for i in range(len(seq)):
    plt.text(poinc[i,0], poinc[i,1], seq[i], bbox=props)
    

plt.xlim([-1,1])
plt.ylim([-1,1])
plt.axis('equal')

#%% Try neural net, out of desperation
dim = 2 # dimension of poincare embedding

n_neg = 5
bsz = 32
nepoch = 300
eta = 1e-3
burnin = 20
c_bi = 10

# D = DenseDataset(X, seq, bsz, n_neg=n_neg, padding=False)
# D = DenseDataset(np.diag(np.diagonal(X,1),1), seq, bsz, n_neg=n_neg, padding=False)
D = SparseGraphDataset(idx, weights, seq, bsz, n_neg=n_neg)
enc = Feedforward([len(seq), dim], [None],
                  encoder=Indicator(len(seq), len(seq)))
# hype = HyperboloidPP(enc)
hype = TangentSpace(enc)
hype.init_weights(torch.tensor(np.arange(len(seq))).long())

optimizer = optim.Adam(hype.parameters(), lr=eta)

train_loss = np.zeros(nepoch)
for epoch in range(nepoch):
    if epoch<=burnin:
        optimizer.param_groups[0]['lr'] = eta/c_bi
    else:
        optimizer.param_groups[0]['lr'] = eta
    
    running_loss = 0
    with tqdm(D, total=D.num_batches, desc='Epoch %d'%epoch, postfix=[dict(loss_=0)]) as looper:
        for i, (n,t) in enumerate(looper):
            
            optimizer.zero_grad()
            
            logprob = -hype.distances(n)
            assert torch.all(logprob==logprob), "Contains NaN"
            
            loss = nn.CrossEntropyLoss()(logprob, t)
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item()
            
            looper.postfix[0]['loss_'] = np.round(running_loss/(i+1), 3)
            looper.update()
            
    # print('Epoch %d, loss=%.3f'%(epoch, running_loss/(i+1)))    
    train_loss[epoch] = running_loss/(i+1)

#%%
wa = hype(torch.tensor(np.arange(len(seq))).long()).detach()

poinc = wa[:,1:]/(1+wa[:,:1])

plt.scatter(poinc[:,0],poinc[:,1])

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

for i in range(len(seq)):
    plt.text(poinc[i,0], poinc[i,1], seq[i], bbox=props)
    

plt.xlim([-1,1])
plt.ylim([-1,1])
plt.axis('equal')

#%%
mds = manifold.MDS(n_components=2)

emb = mds.fit_transform(og[layer,:,:].T)

plt.scatter(emb[:,0], emb[:,1], s=12)

for i in range(len(seq)):
    plt.text(emb[i,0], emb[i,1], seq[i], bbox=props)
 
