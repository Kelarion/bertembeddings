import numpy as np
import scipy.stats as sts
import pickle as pkl
import matplotlib as mpl
import matplotlib.pyplot as plt

# from cycler import cycler
# mpl.rcParams['axes.prop_cycle'] = cycler()

#%%
# tree_type = 'dep'
# tree_type = 'const'
tree_type = 'constdepth'
# tree_type = 'depdepth'
SAVE_DIR = '/home/matteo/Documents/github/bertembeddings/data/bert/'+tree_type+'/'

these_tree_distances = [2,3,4,5]
# these_tree_distances = [1,2,3,4,5,6]
# these_tree_distances = [0,1,2,3,4,5]
these_seq_distances = [1,2,3,4]

CCA = np.zeros((13,len(these_tree_distances),len(these_seq_distances)))
CCA_err = np.zeros((13,len(these_tree_distances),len(these_seq_distances)))

CCA_swp = np.zeros((13,len(these_tree_distances),len(these_seq_distances)))
CCA_swp_err = np.zeros((13,len(these_tree_distances),len(these_seq_distances)))

csim = np.zeros((13,len(these_tree_distances),len(these_seq_distances)))
csim_err = np.zeros((13,len(these_tree_distances),len(these_seq_distances)))

csim_swp = np.zeros((13,len(these_tree_distances),len(these_seq_distances)))
csim_swp_err = np.zeros((13,len(these_tree_distances),len(these_seq_distances)))

avg_dist = np.zeros((13,len(these_tree_distances),len(these_seq_distances)))
avg_dist_err = np.zeros((13,len(these_tree_distances),len(these_seq_distances)))

froeb_dist = np.zeros((13,len(these_tree_distances),len(these_seq_distances)))
froeb_dist_err = np.zeros((13,len(these_tree_distances),len(these_seq_distances)))

nuc_dist = np.zeros((13,len(these_tree_distances),len(these_seq_distances)))
nuc_dist_err = np.zeros((13,len(these_tree_distances),len(these_seq_distances)))

inf_dist = np.zeros((13,len(these_tree_distances),len(these_seq_distances)))
inf_dist_err = np.zeros((13,len(these_tree_distances),len(these_seq_distances)))

orig_dist = np.zeros((13,len(these_tree_distances),len(these_seq_distances)))
orig_dist_err = np.zeros((13,len(these_tree_distances),len(these_seq_distances)))

dim_lin = np.zeros((13,len(these_tree_distances),len(these_seq_distances)))
dim_lin_err = np.zeros((13,len(these_tree_distances),len(these_seq_distances)))

parallel = np.zeros((13,len(these_tree_distances),len(these_seq_distances)))
parallel_err = np.zeros((13,len(these_tree_distances),len(these_seq_distances)))

l2_all = np.zeros((13,len(these_tree_distances),len(these_seq_distances),400))
csim_all = np.zeros((13,len(these_tree_distances),len(these_seq_distances),400))
csim_swp_all = np.zeros((13,len(these_tree_distances),len(these_seq_distances),400))
froeb_all = np.zeros((13,len(these_tree_distances),len(these_seq_distances),400))
nuc_all = np.zeros((13,len(these_tree_distances),len(these_seq_distances),400))
inf_all = np.zeros((13,len(these_tree_distances),len(these_seq_distances),400))

minswap = 400
do_cca=True
for i,t in enumerate(these_tree_distances):
    for j,s in enumerate(these_seq_distances):
        fold = 'tree%d-seq%d/'%(t,s)
        
        try:
            cca_all = pkl.load(open(SAVE_DIR+fold+'cca_coefs.pkl','rb'))
            cca_swp = pkl.load(open(SAVE_DIR+fold+'cca_swp.pkl','rb'))
            do_cca = True
        except FileNotFoundError:
            print('ba!')
            print('%d %d'%(s,t))
            do_cca= False
        
        cosines = np.load(SAVE_DIR+fold+'cosines_subsampled.npy')
        l2 = np.load(SAVE_DIR+fold+'original_distances.npy')
        swap_l2 = np.load(SAVE_DIR+fold+'swap_distances.npy')
        
        swap_idx = np.load(SAVE_DIR+fold+'swap_indices.npy')
        line_idx = np.load(SAVE_DIR+fold+'swap_number.npy')
        
        frob = np.load(SAVE_DIR+fold+'swap_frob_distance.npy')
        nuc = np.load(SAVE_DIR+fold+'swap_nuc_distance.npy')
        inf = np.load(SAVE_DIR+fold+'swap_inf_distance.npy')
        
        dim = np.load(SAVE_DIR+fold+'difference_dimension.npy')
        par = np.load(SAVE_DIR+fold+'difference_parallelism.npy')
        
        nswap = swap_idx.shape[0]
        if nswap<minswap:
            minswap=nswap
        l2_all[:,i,j,:nswap] = l2[:,:nswap]
        # l2_all[:,i,j,:] = swap_l2[:,swap_idx].mean(2)[:,:400]
        csim_swp_all[:,i,j,:nswap] = cosines[:,swap_idx].mean(2)[:,:nswap]
        csim_all[:,i,j,:nswap] = np.array([cosines[:,line_idx==l].mean(1) for l in np.unique(line_idx)[:nswap]]).T
        
        # geometry
        dim_lin[:,i,j] = dim.mean(1)
        dim_lin_err[:,i,j] = dim.std(1)/np.sqrt(dim.shape[1])
        parallel[:,i,j] = par.mean(1)
        parallel_err[:,i,j] = par.std(1)/np.sqrt(par.shape[1])
        
        # CCA
        if do_cca:
            CCA[:,i,j] = np.array([c.mean() for c in cca_all])
            CCA_err[:,i,j] = np.array([c.std()/np.sqrt(len(c)) for c in cca_all])
            
            CCA_swp[:,i,j] = np.array([c.mean() for c in cca_swp])
            CCA_swp_err[:,i,j] = np.array([c.std()/np.sqrt(len(c)) for c in cca_swp])
            
        # matrix norms
        froeb_dist[:,i,j] = frob.mean(0)
        froeb_dist_err[:,i,j] = frob.std(0)/np.sqrt(frob.shape[0])
        froeb_all[:,i,j,:nswap] = frob.T
        
        nuc_dist[:,i,j] = nuc.mean(0)
        nuc_dist_err[:,i,j] = nuc.std(0)/np.sqrt(nuc.shape[0])
        nuc_all[:,i,j,:nswap] = nuc.T
        
        inf_dist[:,i,j] = inf.mean(0)
        inf_dist_err[:,i,j] = inf.std(0)/np.sqrt(inf.shape[0])
        inf_all[:,i,j,:nswap] = inf.T
        
        # pointwise quantitites
        csim_swp[:,i,j] = cosines[:,swap_idx].mean(2).mean(1)
        csim_swp_err[:,i,j] = cosines[:,swap_idx].mean(2).std(1)/np.sqrt(swap_idx.shape[0])
        csim[:,i,j] = cosines.mean(1)
        csim_err[:,i,j] = cosines.std(1)/np.sqrt(cosines.shape[1])
        
        orig_dist[:,i,j] = l2.mean(1)
        orig_dist_err[:,i,j] = l2.std(1)/np.sqrt(l2.shape[1])
        
        avg_dist[:,i,j] = swap_l2.mean(1)
        avg_dist_err[:,i,j] = swap_l2.std(1)/np.sqrt(swap_l2.shape[1])

l2_all = l2_all[...,:minswap]
csim_all = csim_all[...,:minswap]
froeb_all = froeb_all[...,:minswap]
nuc_all = nuc_all[...,:minswap]
inf_all = inf_all[...,:minswap]
csim_swp_all = csim_swp_all[...,:minswap]

#%%
seq_dist = 0

plot_this = CCA
error = CCA_err

# plt.subplot(2,2,1)
for t in range(len(these_tree_distances)):
    plt.plot(np.arange(1,14),plot_this[:,t,seq_dist], marker='o')
    plt.fill_between(np.arange(1,14), 
                      plot_this[:,t,seq_dist]-error[:,t,seq_dist], 
                      plot_this[:,t,seq_dist]+error[:,t,seq_dist],
                      alpha=0.5)

plt.legend(these_tree_distances, title='%s tree distance'%tree_type)
plt.title('For swaps with %d words between'%(seq_dist))
plt.ylabel('Mean CCA (full sentences)')

#%%
seq_dist = 0
whichlayer = 12

plot_this = csim
error = csim_err
plt.subplot(2,2,1)
for t in range(len(these_tree_distances)):
    plt.plot(np.arange(1,14),plot_this[:,t,seq_dist], marker='o')
    plt.fill_between(np.arange(1,14), 
                     plot_this[:,t,seq_dist]-error[:,t,seq_dist], 
                     plot_this[:,t,seq_dist]+error[:,t,seq_dist],
                     alpha=0.5)

plt.legend(these_tree_distances, title='%s tree distance'%tree_type)
plt.title('For swaps with %d words between'%(seq_dist))
plt.ylabel('Average cosine b/t orig ang swap (full sentence)')

plot_this = csim_swp
error = csim_swp_err

plt.subplot(2,2,2)
for t in range(len(these_tree_distances)):
    plt.plot(np.arange(1,14),plot_this[:,t,seq_dist], marker='o')
    plt.fill_between(np.arange(1,14), 
                     plot_this[:,t,seq_dist]-error[:,t,seq_dist], 
                     plot_this[:,t,seq_dist]+error[:,t,seq_dist],
                     alpha=0.5)

plt.legend(these_tree_distances, title='%s tree distance'%tree_type)
plt.title('For swaps with %d words between'%(seq_dist))
plt.ylabel('Average cosine b/t orig ang swap (only swapped)')

# scatter plots
d_tree = np.ones((1,minswap))*np.array(these_tree_distances)[:,None]

plt.subplot(2,2,3)
plot_this = csim_all

rho = [sts.spearmanr(plot_this[l,:,seq_dist,:].flatten(), d_tree.flatten())[0] for l in range(13)]
pval = [sts.spearmanr(plot_this[l,:,seq_dist,:].flatten(), d_tree.flatten())[1] for l in range(13)]
plt.scatter(d_tree.flatten()+np.random.randn(d_tree.flatten().shape[0])*0.1,
            plot_this[-1,:,seq_dist,:].flatten())
plt.xlabel('%s tree distance'%tree_type)
plt.ylabel('Per-swap cosine(full sentence)')
plt.title('Layer %d: spr %.3f, p=%.3f'%(whichlayer+1, rho[whichlayer], pval[whichlayer]))


plt.subplot(2,2,4)
plot_this = csim_swp_all

rho = [sts.spearmanr(plot_this[l,:,seq_dist,:].flatten(), d_tree.flatten())[0] for l in range(13)]
pval = [sts.spearmanr(plot_this[l,:,seq_dist,:].flatten(), d_tree.flatten())[1] for l in range(13)]
plt.scatter(d_tree.flatten()+np.random.randn(d_tree.flatten().shape[0])*0.1,
            plot_this[-1,:,seq_dist,:].flatten())
plt.xlabel('%s tree distance'%tree_type)
plt.ylabel('Per-swap cosine (only swapped)')
plt.title('Layer %d: spr %.3f, p=%.3f'%(whichlayer+1, rho[whichlayer], pval[whichlayer]))


#%%
seq_dist = 0
whichlayer = 11

plot_this = froeb_dist
error = froeb_dist_err
plt.subplot(2,3,1)
for t in range(len(these_tree_distances)):
    plt.plot(np.arange(1,14),plot_this[:,t,seq_dist], marker='o')
    plt.fill_between(np.arange(1,14), 
                     plot_this[:,t,seq_dist]-error[:,t,seq_dist], 
                     plot_this[:,t,seq_dist]+error[:,t,seq_dist],
                     alpha=0.5)

plt.legend(these_tree_distances, title='%s tree distance'%tree_type)
plt.title('For swaps with %d words between'%(seq_dist))
plt.ylabel('Froebenius distance (b/t full sentences)')


plot_this = nuc_dist
error = nuc_dist_err
plt.subplot(2,3,2)
for t in range(len(these_tree_distances)):
    plt.plot(np.arange(1,14),plot_this[:,t,seq_dist], marker='o')
    plt.fill_between(np.arange(1,14), 
                     plot_this[:,t,seq_dist]-error[:,t,seq_dist], 
                     plot_this[:,t,seq_dist]+error[:,t,seq_dist],
                     alpha=0.5)
plt.legend(these_tree_distances, title='%s tree distance'%tree_type)
plt.title('For swaps with %d words between'%(seq_dist))
plt.ylabel('Nuclear distance (b/t full sentences)')


plot_this = inf_dist
error = inf_dist_err
plt.subplot(2,3,3)
for t in range(len(these_tree_distances)):
    plt.plot(np.arange(1,14),plot_this[:,t,seq_dist], marker='o')
    plt.fill_between(np.arange(1,14), 
                     plot_this[:,t,seq_dist]-error[:,t,seq_dist], 
                     plot_this[:,t,seq_dist]+error[:,t,seq_dist],
                     alpha=0.5)
plt.legend(these_tree_distances, title='%s tree distance'%tree_type)
plt.title('For swaps with %d words between'%(seq_dist))
plt.ylabel('Supremum distance (b/t full sentences)')


# scatter plots
d_tree = np.ones((1,minswap))*np.array(these_tree_distances)[:,None]

plt.subplot(2,3,4)
plot_this = froeb_all
rho = [sts.spearmanr(plot_this[l,:,seq_dist,:].flatten(), d_tree.flatten())[0] for l in range(13)]
pval = [sts.spearmanr(plot_this[l,:,seq_dist,:].flatten(), d_tree.flatten())[1] for l in range(13)]
plt.scatter(d_tree.flatten()+np.random.randn(d_tree.flatten().shape[0])*0.1,
            plot_this[whichlayer,:,seq_dist,:].flatten())
plt.xlabel('%s tree distance'%tree_type)
plt.ylabel('Per-swap froebenius distance')
plt.title('Layer %d: spr %.3f, p=%.3f'%(whichlayer+1, rho[whichlayer], pval[whichlayer]))

plt.subplot(2,3,5)
plot_this = nuc_all
rho = [sts.spearmanr(plot_this[l,:,seq_dist,:].flatten(), d_tree.flatten())[0] for l in range(13)]
pval = [sts.spearmanr(plot_this[l,:,seq_dist,:].flatten(), d_tree.flatten())[1] for l in range(13)]
plt.scatter(d_tree.flatten()+np.random.randn(d_tree.flatten().shape[0])*0.1,
            plot_this[whichlayer,:,seq_dist,:].flatten())
plt.xlabel('%s tree distance'%tree_type)
plt.ylabel('Per-swap nuclear distance')
plt.title('Layer %d: spr %.3f, p=%.3f'%(whichlayer+1, rho[whichlayer], pval[whichlayer]))


plt.subplot(2,3,6)
plot_this = inf_all
rho = [sts.spearmanr(plot_this[l,:,seq_dist,:].flatten(), d_tree.flatten())[0] for l in range(13)]
pval = [sts.spearmanr(plot_this[l,:,seq_dist,:].flatten(), d_tree.flatten())[1] for l in range(13)]
plt.scatter(d_tree.flatten()+np.random.randn(d_tree.flatten().shape[0])*0.1,
            plot_this[whichlayer,:,seq_dist,:].flatten())
plt.xlabel('%s tree distance'%tree_type)
plt.ylabel('Per-swap supremum distance')
plt.title('Layer %d: spr %.3f, p=%.3f'%(whichlayer+1, rho[whichlayer], pval[whichlayer]))

#%%
seq_dist = 0

d_tree = np.ones((1,minswap))*np.array(these_tree_distances)[:,None]

plt.plot([sts.spearmanr(froeb_all[l,:,seq_dist,:].flatten(), d_tree.flatten())[0] for l in range(13)],marker='o')
plt.plot([sts.spearmanr(nuc_all[l,:,seq_dist,:].flatten(), d_tree.flatten())[0] for l in range(13)],marker='o')
plt.plot([sts.spearmanr(inf_all[l,:,seq_dist,:].flatten(), d_tree.flatten())[0] for l in range(13)],marker='o')

plt.title('%s tree, swaps %d words apart, over distances %s'%(tree_type, seq_dist, these_tree_distances))
plt.legend(['Froebenius','Nuclear','Supremum'],title='Matrix distances')
plt.xlabel('Layer')
plt.ylabel('Spr.')

#%%
seq_dist = 0

plot_this = dim_lin
error = dim_lin_err

# plt.subplot(2,2,1)
for t in range(len(these_tree_distances)):
    plt.plot(np.arange(1,14),plot_this[:,t,seq_dist], marker='o')
    plt.fill_between(np.arange(1,14), 
                      plot_this[:,t,seq_dist]-error[:,t,seq_dist], 
                      plot_this[:,t,seq_dist]+error[:,t,seq_dist],
                      alpha=0.5)

plt.legend(these_tree_distances, title='%s tree distance'%tree_type)
plt.title('For swaps with %d words between'%(seq_dist))
plt.ylabel('Mean CCA (full sentences)')
