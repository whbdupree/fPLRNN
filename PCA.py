import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import matplotlib.cm as cm
import matplotlib

import argparse
from os.path import join

from utils import data_loader

plt.rcParams.update({
            'font.family':'sans-serif',
            'font.sans-serif':['Liberation Sans'],
            })

tt=np.arange(151)*0.2

def plotPC(ax,wb,lnw,alpha,color):
    for w in wb:
        ax.plot(
            tt, w,
            linewidth = lnw,
            color = color,
            alpha = alpha
        )

def PCs_sorted_trjs( ax, z_, loadings ):
    sorted_loading_idx = np.argsort( loadings )
    sorted_loadings = loadings[ sorted_loading_idx ]
    strong_negative_idx = sorted_loading_idx[ :5]
    strong_positive_idx = sorted_loading_idx[ -5:]
    strong_negative = loadings[ strong_negative_idx]
    strong_positive = loadings[ strong_positive_idx]
    z_strong_positive = z_[...,strong_positive_idx]
    z_strong_negative = z_[...,strong_negative_idx]    
    offset = 0
    doff = 5
    s = 0.7 # scale z relative to doff
    #print('z shape:',z_.shape)
    ## get large negative loaders
    for i in range(len(strong_negative_idx)):
        z = z_strong_negative[...,i] * s
        ax.plot( tt, z[0]+offset,color ='C0')
        ax.plot( tt, z[1]+offset,color ='C1')
        offset+=doff
    offset+=(doff/2)

    ## get large positive loaders
    for i in range(len(strong_negative_idx)):
        z = z_strong_positive[...,i] * s
        ax.plot( tt, z[0]+offset,color ='C0')
        ax.plot( tt, z[1]+offset,color ='C1')
        offset+=doff

    ax.spines[['left', 'right', 'top','bottom']].set_visible(False)    
    ax.tick_params(
        axis='x',
        which='both',
        top=False,
        bottom=False,
        labelbottom=False
    )
    ax.tick_params(
        axis='y',
        left=False,
        right=False,
        labelleft=False,
        labelright=False
    )
    ax.set_xlim([0,30])
    ax.set_xticks([0,10,20,30],[-20,-10,0,10])
    #ax.set_xlabel('Time (s)')
    ax.annotate(
        'High Loading Latent Factors',
        xy = (-0.25,0.15),
        xycoords='axes fraction',
        rotation=90,
        fontsize = 11
    )
    ax.annotate(
        'Negative',
        xy = (-0.15,0.2),
        xycoords='axes fraction',
        rotation=90, color = 'b',
    )
    ax.annotate(
        'Positive',
        xy = (-0.15,0.62),
        xycoords='axes fraction',
        rotation=90, color = 'r',
    )

    
    ax.annotate(
        '',
        xy = (-0.05,0.055),
        xycoords='axes fraction',
        xytext = (-0.05,0.49),
        textcoords='axes fraction',
        arrowprops=dict(arrowstyle='->', color='b',
                        shrinkA=5, shrinkB=5,
                        ),
    )
    ax.annotate(
        '',
        xy = (-0.05,0.95),
        xycoords='axes fraction',
        xytext = (-0.05,0.51),
        textcoords='axes fraction',
        arrowprops=dict(arrowstyle='->', color='r',
                        shrinkA=5, shrinkB=5,
                        ),
    )

    return (strong_negative,strong_positive), (z_strong_negative, z_strong_positive)
    
def loading_distributions( ax, loadings, select_loadings, PCname ):
    for loading in loadings:
        clr = 0#0.7
        ax.plot([loading]*2, [0,1],color=[clr]*3)
    # order is negative, positive
    colors = ['b','r']
    for color,loadings_group in zip(colors,select_loadings):
        for loading in loadings_group:
            ax.plot([loading]*2, [0,1],color = color)

    ax.tick_params(
        axis='y',
        which='both',
        left=False,
        right=False,
        labelleft=False
    )
    #ax.set_xlabel('Loading Coefficients')
    ax.set_title(f'{PCname} Loading Coefficients')
    
def PCs_with_errors( ax1, ax2, w, high_loaders, PCname):
    gray = [0.7]*3    
    #print('w shape:',w.shape)
    wi = w[0]
    wd = w[1]

    z_high_loaders = np.concatenate( high_loaders, axis=2 )
    diff_zhl = np.diff(z_high_loaders,axis=1)
    dd = np.sqrt( np.sum( diff_zhl**2, axis=2 ) )
    ddi = dd[0]
    ddd = dd[1]
    ## set up fignnn
    ax1.tick_params(
        axis='x',
        which='both',
        top=False,
        bottom=False,
        labelbottom=False
    )
    ax2.tick_params(
        axis='x',
        which='both',
        top=False,
        bottom=True,
        labelbottom=True
    )

    left = True
    right = False
    labelleft = True
    labelright = False
    ax1.spines[['right', 'top','bottom']].set_visible(False)
    ax1.tick_params(
        axis='y',
        left=left,
        right=right,
        labelleft=labelleft,
        labelright=labelright
    )

    ax1.set_title(f'State Projection onto {PCname}')
    ax1.plot( [tt[0],tt[-1]],[0,0],'--k',linewidth = 0.5)
    ax1.plot( tt, wi ,'C0')
    ax1.plot( tt, wd ,'C1')
    ax1.set_xlim([0,30])
    ax1.set_ylabel(f'{PCname}')

    ax2.spines[['right', 'top']].set_visible(False)            
    ax2.tick_params(
        axis='y',
        left=left,
        right=right,
        labelleft=labelleft,
        labelright=labelright
    )
    ax2.plot( tt[:-1], ddi/0.2, 'C0' )
    ax2.plot( tt[:-1], ddd/0.2, 'C1' )
    ax2.set_ylabel('Speed')
    ax2.set_yticks( (0,2) )
    ax2.set_xlim( (0,30) )
    ax2.set_xticks( (0,10,20,30), (-20,-10,0,10) )
    ax2.set_xlabel('Time(s)')

def run_pca(z_,ncomp=2):
    uz = np.mean(z_,axis=1,keepdims=True)
    sz = np.std(z_,axis=1,keepdims=True)
    z = (z_ -uz)/sz
    z_shape = z.shape
    L = z_shape[-2]
    U,S,Vh = np.linalg.svd(z,full_matrices = False)
    ev = (S**2) /(L-1)
    ev_tot = np.sum(ev,axis=1,keepdims=True)
    evr = ev / ev_tot

    return U[...,:ncomp]*S[...,None,:ncomp],Vh,evr[...,:ncomp]
    

def main(D,experiment):
    epath = os.path.join( os.getcwd(), 'results',experiment )
    fpath = os.path.join( epath, 'figures' )
    rpath = os.path.join( epath, f'run{D}' )

    zzz = np.load( os.path.join(rpath,'z.npz') )
    z=zzz['testing']
    zs = z.shape
    #print('shape:',zs)
    M = zs[0]
    num_batches = zs[1]
    T = zs[3]
    MAEfile = np.load( os.path.join(rpath,'MAEs.npz') )
    MAE = MAEfile['testing']
    
    NCOMP=10 # number of PCA components    
    ## pca on observations
    _, obs_ = data_loader()
    obs = np.concatenate(obs_,axis=0)
    #print('obs shape:',obs.shape)
    
    u = np.mean( obs , axis=0 )
    s = np.std( obs , axis=0 )
    o = (obs - u)/s
    obsU,obsS,_ = np.linalg.svd( o, full_matrices = False )
    wobs = obsU * obsS # PC projections
    explained_variance = obsS**2/( T+T-1 )
    total_variance = np.sum(explained_variance)
    # explained variance ratio:
    obs_ev = explained_variance / total_variance[...,None]
    obs_top2 = np.sum( obs_ev[:2] )
    print('observation top 2 PCs; explained variance:')
    print(obs_ev[:2])
    print('observation combined PC 1,2 explained variance:')
    print(obs_top2)
    
    ## pca on latent factors
    zcat = z.reshape( M, num_batches*zs[2]*T, D )
    #print('zcat shape:',zcat.shape)    
    PCs_,compv,evr = run_pca( zcat , NCOMP )
    PCs = PCs_.reshape( M, num_batches, zs[2], T, NCOMP )
    np.savez( os.path.join(rpath,'zPC.npz') , **{ 'testing':PCs } )
    #print('PCs shape:',PCs.shape)
    #print( 'evr shape:',evr.shape)
    top2_ev_by_model_instance =  np.sum( evr[:,:2], axis=1 )
    top2 = np.mean( top2_ev_by_model_instance )
    print('LF mean combined PC 1,2 explained variance:')
    print(top2)

    unaltered_PCs = np.copy( PCs )
    idx0=0 # PC1
    idx1=1 # PC2
    PCs_for_plots = PCs.reshape(M*num_batches,zs[2],T,NCOMP)

    ## PC1: clustering to align randomly flipped PCs
    # immediate:
    w00 = PCs_for_plots[:,0,:,idx0] # immediate
    w10 = PCs_for_plots[:,1,:,idx0] # delay
    #print( 'plot PCs shape:',PCs_for_plots.shape)
    clusters = AgglomerativeClustering(n_clusters=2).fit(w00)
    labels = clusters.labels_
    w00[labels==1] = -w00[labels==1] # pick labels==1
    w10[labels==1] = -w10[labels==1] # for visual appeal
    mw00 = np.median( w00, axis=0)
    mw10 = np.median( w10, axis=0)    
    
    ## PC2: clustering to align randomly flipped PCs
    # immediate:
    w01 = PCs_for_plots[:,0,:,idx1] # immediate
    w11 = PCs_for_plots[:,1,:,idx1] # delay
    clusters = AgglomerativeClustering(n_clusters=2).fit(w01)
    labels = clusters.labels_
    w01[labels==0] = -w01[labels==0] # pick labels==0
    w11[labels==0] = -w11[labels==0] # for visual appeal
    mw01 = np.median( w01, axis=0)
    mw11 = np.median( w11, axis=0)    

    ## plot pca results
    fig = plt.figure()
    lnw = 0.25
    alpha = 0.05
    
    Lx = 0.1
    Mx = 0.38
    Rx = 0.69
    y1a = 0.68
    y1b = 0.76

    y2a = 0.39
    y2b = 0.47
    y3 = 0.1

    colW1 = 0.22
    colW2 = 0.2    
    rowH = 0.26
    shortH = 0.18
    shortOffset = 0.07
    
    axL1 = fig.add_axes( (Lx,y1a,colW1,rowH) )
    axM1 = fig.add_axes( (Mx,y1a,colW1,rowH) )
    axR1 = fig.add_axes( (Rx,y1b,colW2,shortH) )

    axL2 = fig.add_axes( (Lx,y2a,colW1,rowH) )
    axM2 = fig.add_axes( (Mx,y2a,colW1,rowH) )
    axR2 = fig.add_axes( (Rx,y2b,colW2,shortH) )

    axL3 = fig.add_axes( (Lx,y3,colW1,rowH) )
    axM3 = fig.add_axes( (Mx,y3,colW1,rowH) )
    axR3 = fig.add_axes( (Rx,y3,colW2,rowH) )

    # plot data PCA
    axL1.plot( tt, wobs[:T,0], color='C0')
    axL1.plot( tt, wobs[T:,0], color='C1')
    axL1.set_ylabel('Observation PCs')

    axL1.set_title('PC1')
    axM1.plot( tt, wobs[:T,1], color='C0')
    axM1.plot( tt, wobs[T:,1], color='C1')
    axM1.set_title('PC2')
    positions = np.arange(NCOMP)+1
    axR1.plot( positions,100*obs_ev[:NCOMP], color='k')
    axR1.plot( positions,100* obs_ev[:NCOMP],'.', color='k')
    
    xlim = [0,30]
    xticks=[0,10,20,30]
    xblank=['','','','']
    axL1.set_xlim(xlim)
    axL1.set_xticks(xticks,xblank)
    axM1.set_xlim(xlim)
    axM1.set_xticks(xticks,xblank)
    axR1.set_xticks((1,3,5,7,9))
    axR1.set_yticks((0,10,20))
    axR1.set_xlabel('Observation PCs')
    axR1.set_ylabel('Explained Var. (%)')
    # plot LF PCAs
    for m in range(M):
        # immediate
        wb0 = PCs[m,:,0,:,idx0]
        wb1 = PCs[m,:,0,:,idx1]
        plotPC( axL2, wb0, lnw, alpha, 'C0' )
        plotPC( axM2, wb1, lnw, alpha, 'C0' )
        # delay
        wb0 = PCs[m,:,1,:,idx0]
        wb1 = PCs[m,:,1,:,idx1]
        plotPC( axL3, wb0, lnw, alpha, 'C1' )
        plotPC( axM3, wb1, lnw, alpha, 'C1' )

    axL2.plot(tt,mw00,'k--',linewidth=0.5)
    axM2.plot(tt,mw01,'k--',linewidth=0.5)
    axL3.plot(tt,mw10,'k--',linewidth=0.5)
    axM3.plot(tt,mw11,'k--',linewidth=0.5)
        
    axR3.plot(mw00,mw01,color='C0')
    axR3.plot(mw10,mw11,color='C1')
        
    axR3.plot(mw00[0],mw01[0],'ok',
                  fillstyle = 'none')                      
    axR3.plot(mw00[-1],mw01[-1],'xk')
    axR3.plot(mw00[100],mw01[100],'s',
                  markeredgecolor = 'k',
                  markerfacecolor = 'C0',)

    axR3.plot(mw10[0],mw11[0],'ok',
                  fillstyle = 'none')                      
    axR3.plot(mw10[-1],mw11[-1],'xk')
    axR3.plot(mw10[100],mw11[100],'s',
                  markeredgecolor = 'k',
                  markerfacecolor = 'C1',)

    ylim1 = [-8,11]
    ylim2 = [-8,10]
    yticks1=[-5,0,5,10]
    yticks2=[-5,0,5,10]

    # PC1 immediate
    axL2.set_ylim(ylim1)
    axL2.set_xlim(xlim)
    axL2.set_xticks(xticks,xblank)
    axL2.set_yticks(yticks1)
    axL2.set_ylabel('Immediate\nLatent Factor PCs')
        
    # PC1 delay
    axL3.set_ylim(ylim1)
    axL3.set_xlim(xlim)
    axL3.set_xticks(xticks,[-20,-10,0,10])
    axL3.set_xlabel('Time (s)')
    axL3.set_yticks(yticks1)
    axL3.set_ylabel('Delay\nLatent Factor PCs')

    # PC2 immediate
    axM2.set_ylim(ylim2)
    axM2.set_xlim(xlim)
    axM2.set_xticks(xticks,xblank)
    axM2.set_yticks(yticks2)

    # PC2 delay
    axM3.set_ylim(ylim2)
    axM3.set_xlim(xlim)
    axM3.set_xticks(xticks,[-20,-10,0,10])
    axM3.set_xlabel('Time (s)')
    axM3.set_yticks(yticks2)

    #
    axR3.set_ylabel('PC2')
    axR3.set_yticks((-3,0,3))
    axR3.set_xlabel('PC1')
    
    ## immediate boxplots
    data = evr[:,:NCOMP]

    axR2.boxplot(
        100*data,
        positions = positions,
        medianprops={'color':'k'},
        showfliers=False
    )
    axR2.set_xticks([1,3,5,7,9],[1,3,5,7,9])
    axR2.set_xlabel('Latent Factor PCs')
    axR2.set_ylabel('Explained Var. (%)')
    this_path = os.path.join( fpath, f'PCA_batch_timeCat_D{D}.pdf' )
    plt.savefig( this_path ) 

    #####################
    # plt strongly loaded latent factors for an exemplar model instance
    m_ = 6 # we pick the seventh best model instance for visualization
    # ^^ this is for visual appeal
    m = np.argsort( MAE[:,-1] )[m_]
    b=0 
    zmb = z[m,b] # model instance m; batch b
    #print('comp batch etc shape:',compv.shape)
    comp = compv[m]

    # these got moved around, and the numerical order
    # of rows may differ from the appearance in the figure.
    # the row number corresponds to:
    # row 1: PC projections
    # row 2: PC condition diffs
    # row 3: sorted latent factors
    # row 4: loading coefficients
    fig1 = plt.figure()
    colW = 0.3
    Lx0 = 0.15
    Rx0 = 0.57
    L1y0 = 0.19
    L2y0 = 0.12
    L3y0 = 0.3
    L4y0 = 0.9
    row1H = 0.06
    row2H = 0.06
    row3H = 0.56
    row4H = 0.03
    ax1L = fig1.add_axes( (Lx0,L1y0,colW,row1H) ) 
    ax2L = fig1.add_axes( (Lx0,L2y0,colW,row2H) ) 
    ax3L = fig1.add_axes( (Lx0,L3y0,colW,row3H) ) 
    ax4L = fig1.add_axes( (Lx0,L4y0,colW,row4H) ) 
    ax1R = fig1.add_axes( (Rx0,L1y0,colW,row1H) , sharey = ax1L ) 
    ax2R = fig1.add_axes( (Rx0,L2y0,colW,row2H) , sharey = ax2L ) 
    ax3R = fig1.add_axes( (Rx0,L3y0,colW,row3H) ) 
    ax4R = fig1.add_axes( (Rx0,L4y0,colW,row4H) ) 
    
    
    ## PC1 loading order plots
    PC1_loadings = comp[0]
    #print('comp shape:',comp.shape)
    PC1_select_loadings,PC1_select_loaders = PCs_sorted_trjs( ax3L, zmb, PC1_loadings )
    ## PC2 loading order plots
    PC2_loadings = comp[1]
    PC2_select_loadings,PC2_select_loaders = PCs_sorted_trjs( ax3R, zmb, PC2_loadings )
    xL_ann = 0.35

    ax3L.annotate(
        '',
        xy = (xL_ann,0.29),
        xycoords='figure fraction',
        xytext = (xL_ann,0.85),
        textcoords='figure fraction',
        arrowprops=dict(arrowstyle='->', color='C2',
                        shrinkA=5, shrinkB=5,
                        alpha = 0.7
                        ),
    )
    xR_ann = 0.77
    ax3R.annotate(
        '',
        xy = (xR_ann,0.29),
        xycoords='figure fraction',
        xytext = (xR_ann,0.85),
        textcoords='figure fraction',
        arrowprops=dict(arrowstyle='->', color='C2',
                        shrinkA=5, shrinkB=5,
                        alpha = 0.7,
                        ),
    )
    loading_distributions( ax4L, PC1_loadings, PC1_select_loadings, 'PC1' )
    loading_distributions( ax4R, PC2_loadings, PC2_select_loadings, 'PC2' )

    wtt = unaltered_PCs[m,b]
    # rename this function
    PCs_with_errors( ax1L, ax2L, wtt[...,0], PC1_select_loaders, 'PC1' )
    PCs_with_errors( ax1R, ax2R, wtt[...,1], PC2_select_loaders, 'PC2' )
    
    ############### next fig
    
    my_cmap1 = 'turbo'
    my_cmap2 = 'hot'
    
    fig2,axs = plt.subplots(2,2,width_ratios=[0.3,0.7])
    ax1 = axs[0,1]

    z0_ = zmb[0] # im
    z1_ = zmb[1] # de
    ijmax0 = np.argsort( np.argmax(np.abs(z0_),axis=0) )
    z0 = np.abs( z0_[:,ijmax0] )
    ijmax1 = np.argsort( np.argmax(np.abs(z1_),axis=0) )
    z1 = np.abs( z1_[:,ijmax1] )
    cmap1 = matplotlib.colormaps[my_cmap1]
    normalizer1 = Normalize( 0 , np.max( np.abs(zmb) ) )
    im1 = cm.ScalarMappable( norm = normalizer1, cmap = cmap1 )


    
    ax1.pcolormesh( z0.T, cmap = cmap1, norm = normalizer1 )
    ax1.set_xticks( [0.5,50.5,100.5,150.5],[-20,-10,0,10] )
    #ax1.set_ylabel('Immediate Choice')        
    ax2 = axs[1,1]
    ax2.pcolormesh( z1.T, cmap = cmap1, norm = normalizer1 )
    ax2.set_xticks( [0.5,50.5,100.5,150.5],[-20,-10,0,10] )
    ax2.set_xlabel( 'Time (s)' )
    #ax2.set_ylabel('Delay Choice')
    PC12 = np.abs( np.vstack((PC1_loadings,PC2_loadings)) )

    cmap2 = matplotlib.colormaps[my_cmap2]
    normalizer2 = Normalize( 0 , np.max( PC12 ) )
    im2 = cm.ScalarMappable( norm = normalizer2, cmap = cmap2 )
    
    ax3 = axs[0,0]
    ax3.pcolormesh( PC12[:,ijmax0].T, cmap = cmap2 )
    ax3.set_ylabel('Immediate Choice')
    ax3.tick_params(
        axis='y',
        left=False,
        right=True,
        labelleft=False,
        labelright=True
    )
    ax3.set_yticks([0,10,20,30],['','','','',])
    ax3.set_xticks([0.5,1.5],[1,2])
    
    ax4 = axs[1,0]
    ax4.pcolormesh( PC12[:,ijmax1].T, cmap = cmap2 )
    ax4.set_ylabel('Delay Choice')
    ax4.tick_params(
        axis='y',
        left=False,
        right=True,
        labelleft=False,
        labelright=True
    )
    ax4.set_yticks([0,10,20,30],['','','','',])
    ax4.set_xticks([0.5,1.5],[1,2])
    
    fig2.colorbar( im1, ax = axs[:,1].ravel().tolist() ,
                   aspect = 30,
                   label = 'Rectified Latent Model Activity' )
    fig2.colorbar( im2, ax = axs[:,0].ravel().tolist() ,
                   location = 'left',
                   aspect = 30,
                   fraction = 0.5,
                   pad = 0.3,
                   label = 'Rectified PC Loading Coefficient' )

    ### write pdf out
    this_path = os.path.join( fpath, f'zPCA_batch_sortedLoadings_timeCat_D{D}.pdf' )    
    with PdfPages( this_path ) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)

    plt.close(fig1)
    plt.close(fig2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_z', metavar='D', type=int, default = 30)    
    args = parser.parse_args()
    D = args.dim_z
    experiments= ['basicRNN_model_size','linearRNN_model_size']

    for experiment in experiments:
        print('experiment:',experiment)
        main(D,experiment)
