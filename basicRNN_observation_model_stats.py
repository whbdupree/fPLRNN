import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import argparse
plt.rcParams.update({
            'font.family':'sans-serif',
            'font.sans-serif':['Liberation Sans'],
            })

nums_clusters = [5,10,20]

def getHist( data, ii ):
    hh = np.zeros(ii+1)
    ud = np.unique( data )
    for u in ud:
        hh[int(u)] = data[ data == u ].size
    return hh

def byWholeCluster(lld,llus,ii,mask = None,maskflag=False):
    lldf = lld.flatten()
    if maskflag:
        mf = mask.flatten()
        lldf = lldf[mf]
    h_data = getHist( lldf, ii )
    mylist = []
    for ll in llus:
        llf = ll.flatten()
        if maskflag:
            mf = mask.flatten()
            llf = llf[mf]
        x = getHist( llf, ii )
        mylist.append(x)
    h_uniform = np.array( mylist )

    return h_data, h_uniform

def make_mask(li):
    L = np.sum(li)
    print('L',L)
    m = np.ones((L,L)).astype(dtype=bool)
    for i in range(li.size):
        sli = np.sum(li[:i])
        sli_n = sli+li[i]
        m[sli:sli_n,sli:sli_n] = False
    return m

def getLikelihood( df ):
    us = df['model_instance'].unique() 
    un = df['nID'].unique() 
    lun = len(un)
    lls =np.zeros((len(nums_clusters),lun,lun),dtype=np.uint16)
    for c,nc in enumerate(nums_clusters):
        cn = f'nc{nc}'
        for s in us:
            sdf = df[ df['model_instance'] == s ]
            uc = np.arange(nc)
            cluster_map = np.array([sdf[cn]==i for i in uc])
            sdf_cn = sdf[cn].values
            logical_index = cluster_map[ sdf_cn[ np.arange(lun)] ]
            lls[c,:,:] += logical_index
    return lls

def run_likelihoods( rpath, D, P ):
    df = pd.read_csv(
        os.path.join( rpath, 'loading_matrix_clusters.csv' )
    )
    ii = len( df['model_instance'].unique() )
    ll_data = getLikelihood( df )
    np.save(
        os.path.join( rpath, 'cluster_likelihoods_data.npy' ),
        ll_data
    )
    
    sgroup = df.groupby('model_instance')
    lls_uniform = []
    futures = []
    with ProcessPoolExecutor() as executor:
        for p in range(P):
            slist = []
            for sm, sg in sgroup:
                sdf = sg.copy(deep=True)
                for nc in nums_clusters:                
                    perm = np.random.permutation( sdf[f'nc{nc}'] )
                    sdf[f'nc{nc}'] = perm
                slist.append(sdf)
            dfnull = pd.concat(slist)
            future = executor.submit( getLikelihood, dfnull )
            futures.append( future )
            
        for p in range(P):
            ll_null = futures[p].result()
            npy_path = f'cluster_likelihoods_null{p}.npy'
            np.save(
                os.path.join( rpath, npy_path ),
                ll_null
            )
            lls_uniform.append(ll_null)

    return ll_data, np.array(lls_uniform), ii

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_z', metavar='D', type=int,default=30)
    args = parser.parse_args()
    D = args.dim_z

    P = 100 
    
    experiment= 'basicRNN_model_size'
    epath = os.path.join( os.getcwd(), 'results',experiment )
    fpath = os.path.join( epath, 'figures' )
    rpath = os.path.join( epath, f'run{D}' )
    sorted_lengths = np.load( os.path.join( rpath, 'sorted_lengths.npy' ) )

    ll_data, lls_uniform,ii = run_likelihoods( rpath, D, P )
    
    results = []
    with ProcessPoolExecutor() as executor:
        futures = []
        for i,nc in enumerate(nums_clusters):
            future = executor.submit(
                byWholeCluster, ll_data[i] , lls_uniform[:,i,:,:], ii
            )
            futures.append( future )
            
        for future in futures:
            hData,hUniform = future.result()
            results.append( (hData, hUniform) )

    mask = make_mask( sorted_lengths )
    nonlocal_results = []
    bWC = partial(byWholeCluster, mask=mask, maskflag=True)
    with ProcessPoolExecutor() as executor:
        futures = []
        for i,nc in enumerate(nums_clusters):
            future = executor.submit(
                bWC, ll_data[i] , lls_uniform[:,i,:,:], ii
            )
            futures.append( future )
            
        for future in futures:
            hData,hUniform = future.result()
            nonlocal_results.append((hData,hUniform))


    ## distribution stats
    my_mask = mask.flatten()
    print('masked entries:',np.sum(my_mask))

    def my_ks( a,b):
        resh = ks_2samp( a,b, method='exact' )
        rt = (resh.statistic,resh.pvalue,resh.statistic_location)
        return rt
    
    with ProcessPoolExecutor() as executor:
        for j in range(3):
            print('num clusters:',nums_clusters[j])
            futures = []
            a = (ll_data[j].flatten())[my_mask]
            for p, llu in enumerate( lls_uniform):
                b = (llu[j].flatten())[my_mask]
                futures.append( executor.submit( my_ks, a, b ) )
            rr = np.zeros( (P,3 ) )
            for p in range(P):
                rr[p] = futures[p].result()
            pmax = np.argmax( rr[:,0]) 
            pmin = np.argmin( rr[:,0])
            print( rr[pmax] )
            print( rr[pmin] )
            
    clr = [ 0.7, 0.7, 0.7 ]
    this_path = os.path.join( fpath, f'cluster_distributions_D{D}.pdf')
    with PdfPages( this_path )as pdf:
        for res,resNL,nc in zip(results,nonlocal_results,nums_clusters):
            h_data,h_uniformP = res
            h_NL,hU_NLP = resNL
            
            pp = np.arange(ii+1)/ii
            
            ax1 = plt.subplot(221)
            for h_uniform in h_uniformP:
                ax1.semilogy(
                    pp,h_uniform,
                    color = clr,
                    linewidth = 2,
            )
            ax1.semilogy(
                pp,h_data,
                color = 'k',
            )
                
            ax1.set_xticks([0,.5,1])
            ax1.set_ylabel('Frequency')
            plt.title(f'No. clusters: {nc}')

            ax3 = plt.subplot(223)
            for hU_NL in hU_NLP:
                ax3.semilogy(
                    pp,hU_NL,
                    color = clr,
                    linewidth = 2,
                )
            ax3.semilogy(
                pp,h_NL,
                color = 'k',
            )
            ax3.set_xticks([0,.5,1])
            ax3.set_ylabel('Frequency\nNon-local')

            
            pdf.savefig()
            plt.close()
