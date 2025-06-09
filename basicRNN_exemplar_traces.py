import numpy as np
import argparse
import os
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({
            'font.family':'sans-serif',
            'font.sans-serif':['Liberation Sans'],
            })

def loader( zpath, file='testing' ):
    data = np.load( zpath )
    return data[file]

def latent_plot(ax,t,z,n):
    ax.plot( t, z[0],
              color = 'C0')
    ax.plot( t, z[1],
              color = 'C1')
    ax.set_title(f'latent {n}')
    ax.set_xlim([t[0],t[-1]])
    
def latent_fig(t,z,b,nn):
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2)
    n1,n2,n3,n4 = nn # which latent factor
    latent_plot( ax1, t, z[m,b,...,n1], n1)
    latent_plot( ax2, t, z[m,b,...,n2], n2)
    latent_plot( ax3, t, z[m,b,...,n3], n3)
    latent_plot( ax4, t, z[m,b,...,n4], n4)
    return fig
    
def neuron_plot(ax,t,x,obs,n):
    ax.plot( t, x[0],'--',
              color = 'C0')
    ax.plot( t, x[1],'--',
              color = 'C1')
    ax.plot( t, obs[0],
              color = 'C0')
    ax.plot( t, obs[1],
              color = 'C1')
    ax.set_title(f'neuron {n}')
    ax.set_xlim([t[0],t[-1]])
    
def neuron_fig(t,x,obs,b,nn):
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2)
    n1,n2,n3,n4 = nn # which neurons
    neuron_plot( ax1, t, x[m,b,...,n1], obs[m,b,...,n1], n1)
    neuron_plot( ax2, t, x[m,b,...,n2], obs[m,b,...,n2], n2)    
    neuron_plot( ax3, t, x[m,b,...,n3], obs[m,b,...,n3], n3)
    neuron_plot( ax4, t, x[m,b,...,n4], obs[m,b,...,n4], n4)
    return fig


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_z', metavar='D', type=int,default = 30)    
    args = parser.parse_args()
    dz = args.dim_z

    experiment= 'basicRNN_model_size'
    epath = os.path.join( os.getcwd(), 'results',experiment )
    fpath = os.path.join( epath, 'figures' )
    rpath = os.path.join( epath, f'run{dz}' )
    z   = loader( os.path.join(rpath,'z.npz')    )
    x   = loader( os.path.join(rpath,'x.npz')    )
    obs = loader( os.path.join(rpath,'obs.npz')  )
    MAE = loader( os.path.join(rpath,'MAEs.npz') )
    T = x.shape[-2]
    tt = np.arange(T)*0.2 - 20
    
    m = np.argsort( MAE[:,-1] )[0] # best model instance

    with PdfPages( os.path.join( fpath,f'exemplar_neuron_traces_D{dz}.pdf')) as pdf:
        for b in range(3):
            for n in range(12):
                nn=np.arange(4)+(n*4)
                fig = neuron_fig(tt,x,obs,b,nn)
                fig.suptitle(f'batch {b}')
                pdf.savefig(fig)
                plt.close()

    with PdfPages( os.path.join( fpath,f'exemplar_latent_traces_D{dz}.pdf')) as pdf:
        for b in range(3):
            for n in range(7):
                nn=np.arange(4)+(n*4)
                fig = latent_fig(tt,z,b,nn)
                fig.suptitle(f'batch {b}')
                pdf.savefig(fig)
                plt.close()
        

    ## let's use neuron 46 in batch 0
    ## let's use latent factor 0 in batch 0

    ax1 = plt.subplot(321)
    ax2 = plt.subplot(325)
    xi = x[m,0,...,46]
    oi = obs[m,0,...,46]
    zi = z[m,0,...,0]
    
    ax1.plot( tt, zi[0],
              color = 'C0')
    ax1.plot( tt, zi[1],
              color = 'C1')
    ax1.set_xlim([tt[0],tt[-1]])
    ax2.plot( tt, xi[0],'--',
              color = 'C0')
    ax2.plot( tt, xi[1],'--',
              color = 'C1')
    ax2.plot( tt, oi[0],
              color = 'C0')
    ax2.plot( tt, oi[1],
              color = 'C1')
    ax2.set_xlim([tt[0],tt[-1]])
    ax2.set_xlabel('Time (s)')
    plt.savefig( os.path.join( fpath, f'exemplar_neuron_latent_D{dz}.pdf') )
    plt.close()
