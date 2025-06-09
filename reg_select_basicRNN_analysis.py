import argparse
import os
import yaml
import time
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import AgglomerativeClustering
from functools import partial

rcParams.update(
    {'font.family':'sans-serif',
     'font.sans-serif':['Liberation Sans'],}
)

def color_boxplots( boxplots, color ):
    for patch in boxplots['boxes']:
        patch.set_color(color)
        patch.set_facecolor(color)        
    for patch in boxplots['medians']:
        patch.set_color('k')
    for item in ['whiskers','caps']:
        # there are two whiskers and two caps.
        # we have to color the "first" and then the "second"
        # of each for each boxplot
        for patch in boxplots[item][::2]:
            patch.set_color(color) # color "first"
        for patch in boxplots[item][1::2]:
            patch.set_color(color) # color "second"
    for patch in boxplots['fliers']:
        patch.set_markeredgecolor(color)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_z', metavar='D', type=int,default=30)
    args = parser.parse_args()
    dz = args.dim_z
    
    experiment= 'basicRNN_reg_select'
    epath = os.path.join( os.getcwd(), 'results',experiment )
    fpath = os.path.join( epath, 'figures' )
    os.makedirs( fpath ,exist_ok=True )

    R = 6

    PCs = [None]*R
    MAEs = [None]*R
    MAE_testing = [None]*R
    CCs_testing = [None]*R
    CCs = [None]*R
    MAE_order = [None]*R
    latent_factors = [None]*R
    reg_pars = [None]*R
    ff=[]
    params = [None]*R
    
    t0 = time.time()
    for i in range(R):
        
        print(dz)
        rpath = os.path.join( epath, f'run_dz{dz}_ir{i}' )
        with open( os.path.join(rpath,'config.yaml'), 'r') as cfile:
            config = yaml.load(cfile,Loader=yaml.Loader)
        dz = config['latent_factors']
        L_Breg = config['L_Breg'] 
        latent_factors[i] = dz
        reg_pars[i] = float( L_Breg ) 
        MAE_file = np.load( os.path.join(rpath,'MAEs.npz' ))
        CCs_file = np.load( os.path.join(rpath,'CCs.npz' ))
        CCs[i] = CCs_file
        CCs_testing[i] = CCs_file['testing']
        MAEs[i] = MAE_file
        MAE = MAE_file['testing'][:,-1]
        MAE_testing[i]=MAE
        MAE_order[i] = np.argsort(MAE)
        #MAE_order[i][0]

        params_file = np.load(os.path.join(rpath,'params.npz'),
                              allow_pickle=True)
        i_params = params_file['testing'].item()
        B = i_params['params']['observation_model']['kernel']
        print(i_params.keys())
        params[i] = B
        print(B.shape)

    print( 'performance plots')
    positions=0.2*np.array([-1,0,1,2,2])
    
    def unpack_CCs(CCs):
        CC_train = CCs['training']
        CC_test = CCs['testing']
        CC_SW = CCs['SW']
        CC_sh5,CC_sh15 = CCs['sh']
        my_mean = partial( np.mean, axis=(1,2,3) )
        return  np.array([my_mean(c) for c in ( CC_train, CC_test, CC_SW, CC_sh5, CC_sh15 ) ])

    this_file = f'model_performance_CC_dz{dz}.pdf'    
    with PdfPages( os.path.join( fpath,this_file)) as pdf:
        fig, (ax1,ax2,ax3) = plt.subplots(nrows=3,ncols=1)
        aR = np.arange(R)
        for i in aR[::-1]:
            print(i)
            BB = params[i]
            Bnorm = np.zeros( 100 )
            for j,B in enumerate( BB ):
                Bnorm[j] = np.linalg.norm( np.hstack(B),ord=1 )
            Biqr =  np.zeros( 100 )
            for j,B in enumerate( BB ):
                Biqr[j] = stats.iqr(B)
            
            print( np.mean( Bnorm ) )


            CCi = unpack_CCs( CCs[i] ) 
            b1 = [ ax1.boxplot( CC,
                                positions = [R-i+pos],
                                showfliers=False,
                                patch_artist=True)
                   for pos,CC in zip(positions[:2]+0.1,CCi[:2]) ]
            
            [color_boxplots( h,c) for h,c in zip(b1, ['r','b'])]
            b2 =  ax2.boxplot( Bnorm,
                               positions = [R-i],
                               #patch_artist=True,
                               showfliers=False,)

            b3 =  ax3.boxplot( Biqr,
                               positions = [R-i],
                               #patch_artist=True,
                               showfliers=False,)

            
        ax1.set_xticks(aR+1,np.log10(reg_pars)[::-1])
        ax1.set_ylabel('Model Instance CC')
        ax1.set_ylim([0.7,1])
        ax2.set_xticks(aR+1,np.log10(reg_pars)[::-1])
        ax2.set_ylabel('Norm ord=1')
        ax3.set_ylabel('B IQR')        
        ax3.set_xlabel('log10( LB )')
        ax3.set_xticks(aR+1,np.log10(reg_pars)[::-1])
        pdf.savefig(fig)
        plt.close(fig)

