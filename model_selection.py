import os
import yaml
import time
import glob
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

def main(experiment):
    
    epath = os.path.join( os.getcwd(), 'results',experiment )
    fpath = os.path.join( epath, 'figures' )
    os.makedirs( fpath ,exist_ok=True )
    run_dirs = glob.glob('run*',root_dir=epath)
    run_dir_ints = [run_dir.split('n')[1] for run_dir in run_dirs]
    run_dir_ints.sort(key=int)
    sorted_run_dirs = ['run'+run_dir_int for run_dir_int in run_dir_ints]
    print(sorted_run_dirs)
    R = len(sorted_run_dirs)
    threshold = 0.05
    PCs = [None]*R
    MAEs = [None]*R
    MAE_testing = [None]*R
    CCs_testing = [None]*R
    CCs = [None]*R
    MAE_order = [None]*R
    latent_factors = [None]*R
    ff=[]
    params = [None]*R
    
    t0 = time.time()
    for i in range(R):
        dz = int(run_dir_ints[i])
        run_dir = sorted_run_dirs[i]
        print(dz)
        rpath = os.path.join( epath, run_dir )
        with open( os.path.join(rpath,'config.yaml'), 'r') as cfile:
            config = yaml.load(cfile,Loader=yaml.Loader)
        dz = config['latent_factors']
        latent_factors[i] = dz
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
        params[i] = params_file['testing'].item()
        

    print( 'performance plots')
    positions=0.2*np.array([-1,0,1,2,2])
    
    def unpack_CCs(CCs):
        CC_train = CCs['training']
        CC_test = CCs['testing']
        CC_SW = CCs['SW']
        CC_sh5,CC_sh15 = CCs['sh']
        my_mean = partial( np.mean, axis=(1,2,3) )
        return  np.array([my_mean(c) for c in ( CC_train, CC_test, CC_SW, CC_sh5, CC_sh15 ) ])
    
    with PdfPages( os.path.join( fpath,'model_performance_CC.pdf')) as pdf:
        fig = plt.figure()
        ax1 = plt.subplot(211)
        for i in range(R):
            dz = latent_factors[i]
            CCi = unpack_CCs( CCs[i] ) # should be 5,100
            b1 = [ ax1.boxplot( CC,
                                positions = [i+pos],
                                showfliers=False,
                                patch_artist=True)
                   for pos,CC in zip(positions[:2]+0.1,CCi[:2]) ]
            
            [color_boxplots( h,c) for h,c in zip(b1, ['r','b'])]
        ax1.set_xticks(range(R),latent_factors)
        ax1.set_ylabel('Model Instance CC')
        ax1.set_xlabel('No. Variables in Latent Model')        
        ax1.set_ylim( [ 0.6, 1.0 ] )
        pdf.savefig(fig)
        plt.close(fig)
        
if __name__ == '__main__':
    experiments= ['basicRNN_model_size','linearRNN_model_size']    
    for experiment in experiments:
        main(experiment)
