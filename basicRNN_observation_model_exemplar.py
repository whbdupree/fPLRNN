import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import argparse
from matplotlib import pyplot as plt
plt.rcParams.update({
            'font.family':'sans-serif',
            'font.sans-serif':['Liberation Sans'],
            })

def run_tsne( BB, tsne_size ):
    print(f'Start TSNE ')
    myt = TSNE(
        n_components=tsne_size,
        init = 'pca',
        metric='cosine',
        learning_rate='auto',
        perplexity=30,
        random_state = 111,
    )
    BBe = myt.fit_transform(BB)
    loss = myt.kl_divergence_
    return BBe,loss

def getBB(Rpath,D):
    itrainv = np.load(f'{Rpath}training_list_D{D}.npy')
    itestv = np.load(f'{Rpath}testing_list_D{D}.npy')
    Bbv = np.load(f'{Rpath}rr_testing_B_D{D}.npz')
    BBv = np.zeros((100,D,2563))
    for m in range(100):
        BBm = BBv[m]
        itrain=itrainv[m]
        itest=itestv[m]
        Bbi=Bbv['training'][m] # i for in sample
        Bbo=Bbv['testing'][m]  # o for out of sample
        
        si = Bbi.shape
        so = Bbo.shape
        Bi = Bbi.reshape( si[0], si[1]*si[2] )
        Bo = Bbo.reshape( so[0], so[1]*so[2] )
        
        BBm[:,itrain] = Bi
        BBm[:,itest[:2050]] = Bo
    return BBv

def loader( zpath, file='testing' ):
    data = np.load( zpath )
    return data[file]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_z', metavar='D', type=int,default=30)
    args = parser.parse_args()
    dz = args.dim_z
    tsne_size = 2

    experiment= 'basicRNN_model_size'
    epath = os.path.join( os.getcwd(), 'results',experiment )
    fpath = os.path.join( epath, 'figures' )
    rpath = os.path.join( epath, f'run{dz}' )
    
    MAE = loader( os.path.join(rpath,'MAEs.npz') )
    m = np.argsort( MAE[:,-1] )[0] # best model instance
    
    params = np.load( os.path.join( rpath, 'params.npz' ), allow_pickle=True )
    Blist = []
    for tname in ['training','testing']:
        B_tname = (params[tname].item())['params']['observation_model']['kernel'][m]
        catB = np.hstack(B_tname)
        Blist.append( catB )

    BB = np.concatenate( Blist, axis=1 )
    BBe, loss = run_tsne( BB.T, tsne_size=tsne_size )
    kmeans = KMeans(n_clusters=10,n_init='auto').fit(BBe)
    ml = kmeans.labels_
    ax = plt.subplot(121)
    for i in range(10):
        idx = ml==i
        ax.plot(BBe[idx,0],BBe[idx,1],'.')
    ax.set_xlabel('tSNE1')
    ax.set_ylabel('tSNE2')
    #ax.set_ylim([-42,42])
    plt.savefig( os.path.join( fpath, f'cluster_exemplar_D{dz}.pdf' ) )
    plt.close()
