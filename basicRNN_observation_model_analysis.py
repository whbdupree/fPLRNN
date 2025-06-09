import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import argparse
nums_clusters=[ 5,10,20 ]

def run_kmeans(model_instance,BB):
    #print(f'Start Kmeans model_instance {model_instance}')
    labels_list = []
    col_names = []
    for i,num_clusters in enumerate(nums_clusters):
        kmeans = KMeans(n_clusters=num_clusters,n_init='auto').fit(BB)
        ml = kmeans.labels_
        labels_list.append(ml)
    col_names = [f'nc{nc}'for nc in nums_clusters]
    col_names.extend(['model_instance']) # list of four strings
    lll = BB.shape[0]
    m_list = [model_instance]*lll
    col_vals = labels_list + [m_list]
    df = pd.DataFrame(
        dict(zip(col_names, col_vals))
    )
    df.index.name = 'nID'
    return df

def getBB(rpath,D):
    idx = np.load( os.path.join(rpath,'idx.npz') )
    params = np.load( os.path.join(rpath,'params.npz'), allow_pickle=True)
    Bi = (params['training'].item())['params']['observation_model']['kernel']
    Bo = (params['testing'].item())['params']['observation_model']['kernel']
    itrain=idx['training']
    itest=idx['testing']

    si = Bi.shape
    so = Bo.shape
    B = np.zeros((100,D,2563))
    for m in range(100):
        mBi = np.moveaxis( Bi[m],0,-2 ).reshape(si[2],si[1]*si[3])
        mBo = np.moveaxis( Bo[m],0,-2 ).reshape(so[2],so[1]*so[3])        
        idxi  = itrain[m]
        idxo  = itest[m]
        B[m,:,idxi] = mBi.T
        B[m,:,idxo] = mBo.T
    #print(B[m,:,idxo].shape)
    #print('mBo shape:',mBo.shape)
    return B

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_z', metavar='D', type=int,default=30)
    args = parser.parse_args()
    
    D = args.dim_z
    tsne_size = 3

    experiment= 'basicRNN_model_size'
    epath = os.path.join( os.getcwd(), 'results',experiment )
    fpath = os.path.join( epath, 'figures' )
    rpath = os.path.join( epath, f'run{D}' )

    B = getBB( rpath,D )
    print(B.shape)

    _df = []
    for m,Bm in enumerate(B):
        _df.append(  run_kmeans( m, Bm.T ) )
    df = pd.concat( _df )
    out_path = os.path.join( rpath, f'loading_matrix_clusters.csv' )
    df.to_csv( out_path )
        
