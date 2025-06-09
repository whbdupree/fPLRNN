import os
import yaml
import time
import glob
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import AgglomerativeClustering
from functools import partial
from scipy.stats import ttest_ind

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
        for patch in boxplots[item][::2]:
            patch.set_color(color) # color "first"
        for patch in boxplots[item][1::2]:
            patch.set_color(color) # color "second"
    for patch in boxplots['fliers']:
        patch.set_markeredgecolor(color)

def marker_placement( boxplots, tstats ):
    '''compute where to place markers above boxplots'''

    boxplots_topwhisker = np.zeros((len(boxplots),2))
    for j,boxplot in enumerate(boxplots):
        whiskers = boxplot['whiskers']
        whiskers_y=np.zeros((2,2))
        whiskers_x=0.
        for i,whisker in enumerate(whiskers):
            xdata,ydata = whisker.get_data()
            whiskers_y[i] = ydata
            whiskers_x = xdata[0] 
        top_whisker = np.max( whiskers_y ) 
        boxplots_topwhisker[j]=(whiskers_x,top_whisker)

    pvalues = [s.pvalue for s in tstats]

    markers=[]
    dy = 0.1
    colors=['g','y','c']
    
    def get_y_coord(pvalue,y,dy):
        y_ = y+dy
        threshold = 0.05 / (3*9)
        if pvalue < threshold:
            return '*',y_
        return None,None
    _,y=boxplots_topwhisker[0]
    x,_=boxplots_topwhisker[1]
    marker, y_ =get_y_coord(pvalues[0],y,2*dy)
    if marker:
        markers.append((marker,x,y_,colors[0]))
    
    x,_=boxplots_topwhisker[3]    
    marker, y_ =get_y_coord(pvalues[1],y,dy)
    if marker:
        markers.append((marker,x,y_,colors[1]))
    # fourth

    marker, y_ =get_y_coord(pvalues[2],y,2*dy)
    if marker:
        markers.append((marker,x,y_,colors[2]))
    return markers

def place_markers(tstats, x_positions):
    colors=('g','y','c')
    y_positions = (2,2,1)    
    threshold = 0.05 / (3*9) 
    sig = tuple( '*' if s.pvalue < threshold else None
                 for s in tstats  )
    print(sig)
    marker_feature_zip = zip(sig, x_positions, y_positions,colors)
    markers = tuple( (s,x,y,c) for s,x,y,c in marker_feature_zip if s ) 
    return markers

if __name__ == '__main__':
    experiment= 'basicRNN_control_experiment'
    epath = os.path.join( os.getcwd(), 'results',experiment )
    fpath = os.path.join( epath, 'figures' )
    os.makedirs( fpath ,exist_ok=True )
    run_dirs = glob.glob('run*',root_dir=epath)
    run_dir_ints = [run_dir.split('n')[1] for run_dir in run_dirs]
    run_dir_ints.sort(key=int)
    sorted_run_dirs = ['run'+run_dir_int for run_dir_int in run_dir_ints]
    print(sorted_run_dirs)
    
    R = len(sorted_run_dirs)
    CCs = [None]*R
    latent_factors = [None]*R
    sim_groups = 10
    t0 = time.time()
    for i in range(R):
        dz = int(run_dir_ints[i])
        run_dir = sorted_run_dirs[i]
        print(dz)
        rpath = os.path.join( epath, run_dir )
        with open( os.path.join(rpath,'sg0_config.yaml'), 'r') as cfile:
            config = yaml.load(cfile,Loader=yaml.Loader)
            dz = config['latent_factors']
        latent_factors[i] = dz
        CCs_files=[]
        for sg in range(sim_groups):
            CCs_file = np.load( os.path.join(rpath,f'sg{sg}_CCs.npz' ))
            CCs_files.append( CCs_file )
        CCs[i] = CCs_files
        
    print( 'performance plots')
    positions=0.2*np.array([-1,0,1,2,2])
    
    def unpack_CCs(CCs):
        CC_train = []
        CC_test = []
        CC_SW = []
        CC_sh5 = []
        CC_sh15 = []
        for sg in range(sim_groups):
            sgCCs=CCs[sg]
            sgCC_train = sgCCs['training']
            CC_train.append( sgCC_train )
            sgCC_test  = sgCCs['testing']
            CC_test.append( sgCC_test )
            sgCC_SW    = sgCCs['SW']
            CC_SW.append( sgCC_SW )            
            sgCC_sh5 , sgCC_sh15 = sgCCs['sh']
            CC_sh5.append( sgCC_sh5 )
            CC_sh15.append( sgCC_sh15 )
        my_mean = partial( np.mean, axis=(1,2,3) )
        CCs_out = np.array(
            [ my_mean(np.concatenate(c))
              for c in ( CC_train, CC_test, CC_SW, CC_sh5, CC_sh15 ) ]
        )
        # CCs_out index 1 vs 2,3,4
        ttest_res = ( ttest_ind( CCs_out[1],CCs_out[2]),
                      ttest_ind( CCs_out[1],CCs_out[3]),
                      ttest_ind( CCs_out[1],CCs_out[4]), )
        return CCs_out, ttest_res

        
    with PdfPages( os.path.join( fpath,'model_control_performance_CC.pdf')) as pdf:
        '''
        fig, (ax1,ax2,ax3) = plt.subplots(
            nrows=3,ncols=1,
            height_ratios=(0.4,0.2,0.4),
        )
        '''
        fig = plt.figure()
        # coords
        x0 = 0.1        
        y0B = 0.1
        y0M = 0.45
        y0T = 0.6
        # spans
        colW = 0.8
        rowHT = 0.37
        rowHM = 0.1
        rowHB = 0.37

        ax1 = fig.add_axes( (x0,y0T,colW,rowHT) ) 
        ax2 = fig.add_axes( (x0,y0M,colW,rowHM) ) 
        ax3 = fig.add_axes( (x0,y0B,colW,rowHB) ) 
        ax2.axis('off')
        for i in range(R):
            dz = latent_factors[i]
            CCi, tstats_i = unpack_CCs( CCs[i] )
            print(tstats_i[2])#.pvalue)

            b1 = [ ax1.boxplot( CC,
                                positions = [i+pos],
                                showfliers=False,
                                patch_artist=True)
                   for pos,CC in zip(positions[:2]+0.1,CCi[:2]) ]
            [color_boxplots( h,c) for h,c in zip(b1, ['r','b'])]
            
            b2 = [ ax3.boxplot( CC,
                                positions = [i+pos],
                                showfliers=False,
                                patch_artist=True)
                   for pos,CC in zip(positions[1:],CCi[1:])]
            [color_boxplots( h,c) for h,c in zip(b2, ['b','g','y','c'])]
            markers = place_markers( tstats_i, i+positions[2:] )
            for markerdata in markers:
                marker,x,y,color= markerdata
                ax2.plot(x,y,marker=marker,color=color)
        
        ax3.set_xticks(range(R),latent_factors)
        ax3.set_ylabel('Model Instance CC')
        ax3.set_xlabel('No. Variables in Latent Model')
        ax3.set_yticks([ 0, 0.25, 0.5, 0.75, 1.0 ],['0','','0.5','','1'] )
        ## we take the xlim from ax3 and use it for ax1 and ax2
        ax1.set_xlim(ax3.get_xlim())        
        ax1.set_xticks(range(R),latent_factors)
        ax1.set_ylabel('Model Instance CC')

        ax2.set_ylim((0,3))
        ax2.set_xlim(ax3.get_xlim())
        
        pdf.savefig(fig)
        plt.close(fig)
