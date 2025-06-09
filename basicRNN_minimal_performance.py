import numpy as np
import argparse
import os
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({
            'font.family':'sans-serif',
            'font.sans-serif':['Liberation Sans'],
            })

def loader( zpath  ):
    data = np.load( zpath )
    return data['training'],data['testing']

def getCC(a,b):
    return np.corrcoef(a,b)[0,1]

def color_boxplots(boxplots,colors):
    for patch,color in zip(boxplots['boxes'],colors):
        patch.set_color(color)
        patch.set_facecolor(color)        
    for patch,color in zip(boxplots['medians'],colors):
        patch.set_color('k')
    for item in ['whiskers','caps']:
        # there are two whiskers and two caps.
        # we have to color the "first" and then the "second"
        # of each for each boxplot
        for patch,color in zip(boxplots[item][::2],colors):
            patch.set_color(color) # color "first"
        for patch,color in zip(boxplots[item][1::2],colors):
            patch.set_color(color) # color "second"
    for patch,color in zip(boxplots['fliers'],colors):        
        patch.set_markeredgecolor(color)

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_z', metavar='D', type=int,default = 30)    
    args = parser.parse_args()
    dz = args.dim_z

    experiment= 'basicRNN_model_size'
    epath = os.path.join( os.getcwd(), 'results',experiment )
    fpath = os.path.join( epath, 'figures' )
    rpath = os.path.join( epath, f'run{dz}' )

    ## beware
    # x_ is a tuple containing training and testing predicted obs
    # obs_ is a tuple containing training and testing obs    
    x_   = loader( os.path.join(rpath,'x.npz')    )
    obs_ = loader( os.path.join(rpath,'obs.npz')  )
    _,MAE = loader( os.path.join(rpath,'MAEs.npz') )
    
    m = np.argsort( MAE[:,-1] )[0] # best model instance

    group=['Training','Testing']
    colors = ['r','b']
    bpw=0.5
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(133)    
    this_path = os.path.join(fpath, f'exemplar_performanceD{dz}.pdf' )
    with PdfPages( this_path ) as pdf:
        for l, foo in enumerate( zip(x_,obs_,group,colors) ):

            x,obs,g,c = foo
            xm = x[m]
            om = obs[m]
            xs = xm.shape
            num_batches = xs[0]
            num_neurons = xs[-1]
            CCs_by_neuron = np.zeros((num_batches,2,num_neurons))
            for i in range(num_batches):
                for j in range(2):
                    for k in range( num_neurons):
                        CCs_by_neuron[i,j,k] = getCC( xm[i,j,:,k],
                                                      om[i,j,:,k] )

            mean_CCs = np.mean(CCs_by_neuron,axis=1).ravel()
            print( g, np.median( mean_CCs ) )
            h1 = ax1.boxplot(
                mean_CCs,
                notch=True,
                widths=bpw,
                positions=[l+1],
                tick_labels=[g],
                patch_artist = True,
                showfliers=False,
            )

            color_boxplots( h1, c)
        ax1.set_yticks([0.7,0.8,0.9,1])
        ax1.set_title('CC')
    
        for l, foo in enumerate( zip(x_,obs_,group,colors) ):
            x,obs,g,c = foo
            xm = x[m]
            om = obs[m]
            xs = xm.shape
            num_batches = xs[0]
            num_neurons = xs[-1]
            MAE_by_neuron_by_context = np.zeros((num_batches,2,num_neurons))
            for i in range(num_batches):
                for j in range(2):
                    for k in range( num_neurons):
                        xijk = xm[i,j,:,k]
                        oijk = om[i,j,:,k]
                        mae_ijk = np.mean(np.abs( xijk - oijk ) )
                        MAE_by_neuron_by_context[i,j,k] = mae_ijk
                            
                                                      
            MAE_by_neuron = np.mean( MAE_by_neuron_by_context, axis=1 ).ravel()
            print( g, np.median( MAE_by_neuron_by_context ) )

            h1 = ax2.boxplot(
                MAE_by_neuron,
                notch=True,
                widths=bpw,
                positions=[l+1],
                tick_labels=[g],
                patch_artist = True,
                showfliers=False,
            )
        
            color_boxplots( h1, c)
        ax2.set_title('MAE')
        ax2.set_yticks([0,0.05,0.1,0.15,0.2])
        pdf.savefig( )
        plt.close()
