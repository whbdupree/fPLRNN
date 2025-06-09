import os
import time
import yaml
import numpy as np
import optax
import jax
from jax import random, tree_util, vmap
from jax import numpy as jnp
from flax import linen as nn

from utils import data_loader, organize_data
from optimization import modelInstanceRNN, modelInstanceObs, shuffle

import fPLRNN


    
def main(data, experiment, Net):
    counts, obs_full = data
    seed = 111    
    num_instances = 100
    batch_size = 50
    test_batches = 41 
    train_batches = 10

    S1 = 7000
    S2 = 3000

    l1_ratio_B = 1
    L_Breg = 0.001

    dz_s =[10,20,30,40,50]

    vectorizedDense = nn.vmap(
        nn.Dense,
        in_axes = 0, out_axes = 0,
        variable_axes={'params':0},
        split_rngs={'params':True}
    )
    observation_model= vectorizedDense(
        batch_size,
        use_bias=False,
        name = 'observation_model'
    ) 
    
    hyperparameters = ( l1_ratio_B, L_Breg )
    train_size = batch_size*train_batches
    test_size = batch_size*test_batches
    key = random.PRNGKey(seed) 
    epochs = S1, S2
    for dz in dz_s:

        t0 = time.time()
        key, *skeys = random.split(key,num_instances+1)
        vod = vmap( organize_data, in_axes=(0,None,None,None,None,None,None) )
        obs_train_v, itrain_v, obs_test_v, itest_v = vod(
            jnp.array(skeys),
            obs_full, train_size, test_size,
            batch_size, train_batches, test_batches
        )

        ## training and testing passes; includes scramble W control experiment
        key, *skeys = random.split(key,num_instances+1)
        vmiRNN = vmap( modelInstanceRNN, in_axes=(0,0,0,None,None,None,None) )
        training_out, testing_out, SW_out  = vmiRNN(
            jnp.array(skeys),
            obs_train_v, obs_test_v,
            Net,dz, epochs, hyperparameters
        )
        otvs = obs_test_v.shape
        T = otvs[3]

        ## shuffle 5, suffle 15 control experiments
        sh5_idx = jnp.stack(
            [ shuffle( T, 5 )
              for i in range( num_instances ) ]
        )
        sh15_idx = jnp.stack(
            [ shuffle( T, 15 )
              for i in range( num_instances ) ]
        )
        sh_idx = jnp.stack( (sh5_idx,sh15_idx))
        z_test_v = testing_out[1]
        key, *skeys = random.split(key,num_instances+1)
        # vectorize over sh5,sh15; vectorize over model instances
        vmiObs = vmap(
            vmap( modelInstanceObs, in_axes=(0,0,0,None,None,None,0,None)),
            in_axes=( None, None, None, None, None, None, 0, None )
        )
        sh_out = vmiObs(
            jnp.array(skeys), obs_test_v,z_test_v, observation_model,
            dz, epochs, sh_idx, hyperparameters
        )
        
        ## print some courtesy output so we can monitor the progress
        print('')
        print(experiment)
        print(f'dz: , {dz}')
        CCs_train = training_out[4]
        CCs_test = testing_out[4]
        CCs_SW = SW_out[4]
        CCs_sh5= sh_out[4][0]
        CCs_sh15= sh_out[4][1]
        print('median CCs: (first ten)') # we report the mean
        print( np.median( CCs_train, axis = (1,2,3))[:10])
        print( np.median( CCs_test, axis = (1,2,3))[:10])
        print( np.median( CCs_SW, axis = (1,2,3))[:10])
        print( np.median( CCs_sh5, axis = (1,2,3))[:10])
        print( np.median( CCs_sh15, axis = (1,2,3))[:10])
        MAEs_train = training_out[3]
        MAEs_test = testing_out[3]
        MAEs_SW = SW_out[3]
        MAEs_sh5 = sh_out[3][0]
        MAEs_sh15 = sh_out[3][1]        
        print( 'MAEs: (first ten)')
        print( MAEs_train[:10,-1] )
        print( MAEs_test[:10,-1] )
        print( MAEs_SW[:10,-1] )
        print( MAEs_sh5[:10,-1] )
        print( MAEs_sh15[:10,-1] )
        print( 'elapsed time:',time.time()-t0 )

        ## write out results
        config_dict ={
            'optimization_epochs': [S1,S2],
            'batch_counts': [train_batches,test_batches],
            'latent_factors': int(dz),
            'l1_ratio_B': l1_ratio_B,
            'L_Breg': L_Breg,
        }
        rpath_relative = ('results',experiment,f'run{int(dz)}')
        rpath = os.path.join(os.getcwd(),*rpath_relative)
        os.makedirs(rpath,exist_ok=True)
    
        filestr = os.path.join(rpath,'config.yaml')    
        with open(filestr, 'w') as file:
            yaml.dump(config_dict,file)
        
        np.savez( os.path.join(rpath,'x.npz') ,
                  **{ 'training':training_out[0], 'testing':testing_out[0],
                      'SW':SW_out[0], 'sh':sh_out[0]} )
        np.savez( os.path.join(rpath,'z.npz') ,
                  **{ 'training':training_out[1], 'testing':testing_out[1],
                      'SW':SW_out[1], 'sh':sh_out[1]} )
        np.savez( os.path.join(rpath,'losses.npz') ,
                  **{ 'training':training_out[2], 'testing':testing_out[2],
                      'SW':SW_out[2], 'sh':sh_out[2]} )
        np.savez( os.path.join(rpath,'MAEs.npz') ,
                  **{ 'training':training_out[3], 'testing':testing_out[3],
                      'SW':SW_out[3], 'sh':sh_out[3]} )
        np.savez( os.path.join(rpath,'CCs.npz') ,
                  **{ 'training':training_out[4], 'testing':testing_out[4],
                      'SW':SW_out[4], 'sh':sh_out[4]} )
        np.savez( os.path.join(rpath,'params.npz') ,
                  **{ 'training':training_out[5], 'testing':testing_out[5],
                      'SW':SW_out[5], 'sh':sh_out[5]} )
        np.savez( os.path.join(rpath,'obs.npz') ,
                  **{ 'training':obs_train_v,'testing':obs_test_v } )
        np.savez( os.path.join(rpath,'idx.npz') ,
                  **{ 'training':itrain_v, 'testing':itest_v} )
        
        np.save(os.path.join(rpath,'sorted_lengths.npy'),counts)
        
if __name__ == '__main__':
    Nets = [fPLRNN.nets.PLRNNet, fPLRNN.nets.LRNNet]
    experiments= ['basicRNN_model_size','linearRNN_model_size']    
    data = data_loader()

    for experiment,Net in zip( experiments, Nets ):
        main( data, experiment, Net)
