import os
import time
import yaml
import numpy as np
import optax
import jax
import argparse
from jax import random, tree_util, vmap
from jax import numpy as jnp
from flax import linen as nn

from utils import data_loader, organize_data
from optimization import modelInstanceRNN, modelInstanceObs, shuffle

import fPLRNN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dz_idx', metavar='D', type=int, default = 0)
    parser.add_argument('--sim_group', metavar='G', type=int, default = 0)
    args = parser.parse_args()
    dz_idx = args.dz_idx
    sg = args.sim_group

    
    counts, obs_full = data_loader()

    seed = 22222

    # In this script, I break simulations up into groups
    # of 10 model instances. The size of these groups
    # are tuned such that the group of largest models
    # (dz = 1000) will fit in the memory of my device.
    # I use this command:
    # for j in {0..8}; do time for i in {0..9}; do python control_experiments_basicRNN_optimization.py --dz_idx $j --sim_group $i; done; done
    
    sg_size = 10 # we are going to do 10 sim groups each of size 10
    # which gives us a total of 100 model instances
    
    batch_size = 50
    test_batches = 41 
    train_batches = 10
    S1 = 7000
    S2 = 3000
    l1_ratio_B = 1
    L_Breg = 0.001

    key = random.PRNGKey(seed)
    # these are the values for dz that we expect to use
    dz_vals = [2,5,10,20,50,100,200,500,1000]
    dz_count = len(dz_vals)
    # so we set dz_count to reflect how many we will use
    dz_keys = random.split(key,dz_count)
    # and then we pick out only the dz value and the rng key
    # that we will use right now
    key = dz_keys[dz_idx]
    dz = dz_vals[dz_idx]
    # now we need to repeat this for the model instances
    # in this simulation group
    all_sg_keys = random.split( key,sg_size)
    # pick out rng keys for simulation group that
    # we will evaluate right now
    key = all_sg_keys[sg] 


    experiment = 'basicRNN_control_experiment'
    
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
    ) # becomes specific to dz at .init time
    
    hyperparameters = ( l1_ratio_B, L_Breg )
    train_size = batch_size*train_batches
    test_size = batch_size*test_batches

    epochs = S1, S2
    
    key,*skeys =  random.split(key,sg_size+1)
    vod = vmap( organize_data, in_axes=(0,None,None,None,None,None,None) )
    obs_train_v, itrain_v, obs_test_v, itest_v = vod(
        jnp.array(skeys),
        obs_full, train_size, test_size,
        batch_size, train_batches, test_batches
    )
    ## training and testing passes; includes scramble W control experiment
    key, *skeys = random.split(key,sg_size+1)
    vmiRNN = vmap( modelInstanceRNN, in_axes=(0,0,0,None,None,None,None) )
    training_out, testing_out, SW_out  = vmiRNN(
        jnp.array(skeys), obs_train_v, obs_test_v,
        fPLRNN.nets.PLRNNet,
        dz, epochs, hyperparameters
    )
    otvs = obs_test_v.shape
    T = otvs[3]
    ## shuffle 5, suffle 15 control experiments
    sh5_idx = jnp.stack(
        [ shuffle( T, 5 )
          for i in range( sg_size ) ]
    )
    sh15_idx = jnp.stack(
        [ shuffle( T, 15 )
          for i in range( sg_size ) ]
    )
    sh_idx = jnp.stack( (sh5_idx,sh15_idx))
    z_test_v = testing_out[1]
    key, *skeys = random.split(key,sg_size+1)
    # vectorize over sh5,sh15; vectorize over model instances
    vmiObs = vmap(
        vmap( modelInstanceObs, in_axes=(0,0,0,None,None,None,0,None)),
        in_axes=( None, None, None, None, None, None, 0, None )
    )
    sh_out = vmiObs(
        jnp.array(skeys), obs_test_v,z_test_v, observation_model,
        dz, epochs, sh_idx, hyperparameters
    )
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
    
    filestr = os.path.join(rpath,f'sg{sg}_config.yaml')    
    with open(filestr, 'w') as file:
        yaml.dump(config_dict,file)
        
    np.savez( os.path.join(rpath,f'sg{sg}_x.npz') ,
              **{ 'training':training_out[0], 'testing':testing_out[0],
                  'SW':SW_out[0], 'sh':sh_out[0]} )
    np.savez( os.path.join(rpath,f'sg{sg}_z.npz') ,
              **{ 'training':training_out[1], 'testing':testing_out[1],
                  'SW':SW_out[1], 'sh':sh_out[1]} )
    np.savez( os.path.join(rpath,f'sg{sg}_losses.npz') ,
              **{ 'training':training_out[2], 'testing':testing_out[2],
                  'SW':SW_out[2], 'sh':sh_out[2]} )
    np.savez( os.path.join(rpath,f'sg{sg}_MAEs.npz') ,
              **{ 'training':training_out[3], 'testing':testing_out[3],
                  'SW':SW_out[3], 'sh':sh_out[3]} )
    np.savez( os.path.join(rpath,f'sg{sg}_CCs.npz') ,
              **{ 'training':training_out[4], 'testing':testing_out[4],
                  'SW':SW_out[4], 'sh':sh_out[4]} )
    np.savez( os.path.join(rpath,f'sg{sg}_params.npz') ,
              **{ 'training':training_out[5], 'testing':testing_out[5],
                  'SW':SW_out[5], 'sh':sh_out[5]} )
    np.savez( os.path.join(rpath,f'sg{sg}_obs.npz') ,
              **{ 'training':obs_train_v,'testing':obs_test_v } )
    np.savez( os.path.join(rpath,f'sg{sg}_idx.npz') ,
              **{ 'training':itrain_v, 'testing':itest_v} )
