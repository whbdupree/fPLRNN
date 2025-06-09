import os
import yaml
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from functools import partial
rcParams.update(
    {'font.family':'sans-serif',
     'font.sans-serif':['Liberation Sans'],}
)

def run_pca(z_,ncomp=2):
    uz = np.mean(z_,axis=1,keepdims=True)
    sz = np.std(z_,axis=1,keepdims=True)
    z = (z_ -uz)/sz
    z_shape = z.shape
    #print('z shape:',z_shape)
    T = z_shape[-2]
    U,S,Vh = np.linalg.svd(z,full_matrices = False)
    ev = (S**2) /( T-1)
    ev_tot = np.sum(ev,axis=1,keepdims=True)
    evr = ev / ev_tot
    return U[...,:ncomp]*S[...,None,:ncomp],evr[...,:ncomp]

def process(rpath,ncomp = 2):
    ## organize data and call run_pca
    z_file = np.load( os.path.join(rpath,'z.npz' ))
    z = z_file['testing']
    zs = z.shape
    zcat = z.reshape( zs[0], zs[1]*zs[2]*zs[3], zs[4] )
    PCs,evr = run_pca( zcat , ncomp)
    return PCs.reshape( zs[0],zs[1],zs[2],zs[3],ncomp), evr, z

def exemplar_PCsMN( PCs,  evr, z,m=0,n=1 ):
    pc_shape = PCs.shape
    T = pc_shape[2]
    #print(pc_shape)
    
    fig2,((axT0,axT1),(axB0,axB1)) = plt.subplots(nrows=2,ncols=2)
    for i in range( pc_shape[0] ):
        axT0.plot(PCs[i,0,:,m],color='C0')
        axB0.plot(PCs[i,1,:,m],color='C1')
        axT1.plot(PCs[i,0,:,n],color='C0')
        axB1.plot(PCs[i,1,:,n],color='C1')
    lim_coeff = .05
    PC1range = np.array( ( np.min(PCs[...,m]),np.max(PCs[...,m]) ) )
    PC2range = np.array( ( np.min(PCs[...,n]),np.max(PCs[...,n]) ) )
    PC1d = np.diff(PC1range)
    PC2d = np.diff(PC2range)
    #print('range:',PC1range)
    #print('diff:',PC1d)
    PC1a = lim_coeff*PC1d
    PC2a = lim_coeff*PC2d     
    PC1lims = ( PC1range[0] - PC1a, PC1range[1] + PC1a )
    PC2lims = ( PC2range[0] - PC2a, PC2range[1] + PC2a )
    
    axT0.set_ylim(PC1lims)
    axT1.set_ylim(PC2lims)
    axT0.set_xticks((0,50,100,150),(-20,-10,0,10))
    axT1.set_xticks((0,50,100,150),(-20,-10,0,10))
    axT0.set_title(f'PC{m+1}')
    axT1.set_title(f'PC{n+1}')
    axT0.set_ylabel('Immediate')
    
    axB0.set_ylim(PC1lims)
    axB1.set_ylim(PC2lims)
    axB0.set_xticks((0,50,100,150),(-20,-10,0,10))
    axB1.set_xticks((0,50,100,150),(-20,-10,0,10))
    axB0.set_xlabel( 'Time (s)' )
    axB1.set_xlabel( 'Time (s)' )
    axB0.set_ylabel('Delay')

    ############### next figure
    
    fig3 = plt.figure()
    subfigs = fig3.subfigures(
        nrows = 3, ncols = 1,
        height_ratios=(0.4,0.4,0.2), 
    )
    axs = [ subfig.subplots( 1, 2, gridspec_kw={'wspace':0.3} ) 
            for subfig in subfigs[:2] ]
    ((axT2,axT3),(axB2,axB3)) = axs
    
    sf3 = subfigs[2]
    # these "blank" axes are a dirty and quick
    # solution to precisely control the position 
    # of the colorbar.
    (blank1,ax_sf3,blank2,blank3) = sf3.subplots(4,1)
    blanks = (blank1,blank2,blank3)
    [blank_axis.axis('off') for blank_axis in blanks]

    def plot_colored_segments(
            pcs,
            alt_pcs,
            dd,
            ax1,
            ax2,
            lw1 = 1.2,
            lw2 = 0.5,
    ):
        # intentionaly loop over ii thrice.
        # first, we draw the "alternate PCs" in grey.
        # then we draw the PCs of interest just plain black.
        # exactly on top of them, we draw those PCs with their
        # color-coded speed and with slightly smaller line
        # width compared to the black version of those PCs.
        # the contrast of bright color on black background
        # helps the colors pop out. finally, we plot markers
        # on top of everything to indicate the time of 
        # the choice.
        ii = pc_shape[0]
        # draw alternate PCs in grey
        for i in range( ii ):
            x_,y_ = alt_pcs[i,:,m],alt_pcs[i,:,n]
            g = 0.65
            ax1.plot(
                x_, y_,
                color = (g,)*3,
                linewidth=lw1,
            )
        # draw PCs of interest in black
        for i in range( ii ):            
            x,y = pcs[i,:,m],pcs[i,:,n]
            d=dd[i]
            ax1.plot(
                x, y,
                color = 'k',
                linewidth=lw1,
            )
            ax2.plot(
                np.arange( d.size ),
                d,
                color = 'k',
                linewidth=lw1,
            )
        # draw PCs of interest again in color
        for i in range( ii ):
            x,y = pcs[i,:,m],pcs[i,:,n]
            d = dd[i]
            for j in range( x.size-1 ):
                xx=(x[j],x[j+1])
                yy=(y[j],y[j+1])
                dj = d[j]
                ax1.plot(
                    xx, yy,
                    linewidth = lw2,
                    color=cmap(norm( dj )),
                )
            for j in range( x.size-2 ):                
                jj = (j,j+1)
                dj = d[j]
                djj = (dj,d[j+1])
                ax2.plot(
                    jj,djj,
                    linewidth = lw2,
                    color=cmap(norm( dj )),
                )
        # draw markers to indicate time of choice
        for i in range( ii ):
            x,y = pcs[i,:,m],pcs[i,:,n]
            cx,cy = x[100],y[100]
            ax1.plot( cx,cy, 'o',
                      markersize = 3,
                      markeredgecolor = 'k',
                      markerfacecolor = 'm',
            )
            # get phase point at time of choice
        ax2.plot( (100,100),(vmin,vmax),'m',alpha=0.5)
        ax2.set_yticks((0,1,2,3,4,5))
    diffz = np.diff(z,axis=2)
    dd = np.sqrt( np.sum( diffz**2, axis=3 ) )
    dz0,dz1 = np.squeeze( np.split( dd,2,axis=1) )
    for_extrema = np.concatenate( (dz0,dz1) )
    vmax = np.max( for_extrema )/0.2
    vmin = np.min( for_extrema )/0.2
    speed_lims = (0,vmax*1.04)
    
    cmap = mpl.cm.turbo
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    plot_colored_segments( PCs[:,0],PCs[:,1], dz0/0.2, axT2, axT3 )
    plot_colored_segments( PCs[:,1],PCs[:,0], dz1/0.2, axB2, axB3 )
    
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    plt.colorbar(
        mappable,
        cax = ax_sf3,
        location = 'bottom',
        fraction = 0.08,
        label = 'Latent Factor Speed',
        ticks = (0,1,2,3,4,5),
    )
    ax_sf3.set_xlim((0,vmax))
    speed_lims = (0,vmax*1.04)
    axT2.set_xlim(PC1lims)
    axT2.set_ylim(PC2lims)
    axB2.set_xlim(PC1lims)
    axB2.set_ylim(PC2lims)
    axT3.set_xlim((0,150))
    axT3.set_ylim(speed_lims)
    axB3.set_xlim((0,150))    
    axB3.set_ylim(speed_lims)
    
    axT2.set_ylabel(f'PC{n+1}')        
    axT2.set_xlabel(f'PC{m+1}')
    axB2.set_ylabel(f'PC{n+1}')
    axB2.set_xlabel(f'PC{m+1}')
    
    axT3.set_ylabel(f'Latent Factor Speed')
    axT3.set_xlabel(f'Time (s)')
    axT3.set_xticks((0,50,100,150),(-20,-10,0,10))
    axB3.set_ylabel(f'Latent Factor Speed')
    axB3.set_xlabel(f'Time (s)')
    axB3.set_xticks((0,50,100,150),(-20,-10,0,10))

    subfigs[0].suptitle('Immediate Choice',fontweight='bold')
    subfigs[1].suptitle('Delay Choice',fontweight='bold')
    return fig2,fig3
    

if __name__ == '__main__':
    ncomp = 10
    experiment= 'basicRNN_model_size'
    epath = os.path.join( os.getcwd(), 'results',experiment )
    fpath = os.path.join( epath, 'figures' )
    os.makedirs( fpath ,exist_ok=True )

    dz_s = [30,]
    R = len(dz_s)

    PCs = [None]*R
    MAEs = [None]*R
    MAE_testing = [None]*R
    CCs_testing = [None]*R
    CCs = [None]*R
    MAE_order = [None]*R
    latent_factors = [None]*R
    EVR = [None]*R
    z = [None]*R

    for i,dz_ in enumerate(dz_s):
        dz = int(dz_)
        print('latent factors: ',dz)
        rpath = os.path.join( epath, f'run{dz}' )
        
        # load and organize data for figs
        with open( os.path.join(rpath,'config.yaml'), 'r') as cfile:
            config = yaml.load(cfile,Loader=yaml.Loader)
        dz = config['latent_factors']
        latent_factors[i] = dz
        MAE_file = np.load( os.path.join(rpath,'MAEs.npz' ))
        CCs_file = np.load( os.path.join(rpath,'CCs.npz' ))
        CCs_testing[i] = CCs_file['testing']
        MAEs[i] = MAE_file
        MAE = MAE_file['testing'][:,-1]
        MAE_testing[i]=MAE
        MAE_order[i] = np.argsort(MAE)
        
        # here we run the pca
        PCs[i],EVR[i],z[i] = process( rpath, ncomp = ncomp )

        
    m = 0
    n = 1
    for i in range(R): 
        dz = latent_factors[i]
        my_path = os.path.join( fpath , f'exemplar_dz{dz}_projections_PC{m+1}_PC{n+1}.pdf' )
        with PdfPages( my_path ) as pdf:
            for j in range(10):
                MAE_idx = MAE_order[i][j]
                MAEj = MAE_testing[i][MAE_idx]
                EVRj = EVR[i][MAE_idx]
                CCsj = np.mean( CCs_testing[i][MAE_idx] )
                these_PCs = PCs[i][MAE_idx]
                zj = z[i][j]
                #print('zi shape:',z[i].shape)
                #print('zj shape:',zj.shape)
                fig2,fig3 = exemplar_PCsMN(
                    these_PCs, EVRj, zj, m=m, n=n,
                )
                round_MAE = np.format_float_positional(MAEj,4)
                round_CC =  np.format_float_positional(CCsj,4)
                suptitle = f'Model ID: {MAE_idx};  MAE: {round_MAE};  CC: {round_CC}'
                fig2.suptitle( suptitle )
                #fig3.suptitle( suptitle )                    
                pdf.savefig(fig2)
                pdf.savefig(fig3)                
                plt.close(fig2)
                plt.close(fig3)
