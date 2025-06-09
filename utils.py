import glob
from jax import numpy as jnp
from jax import random
import numpy as np
from scipy import io
import os

def organize_data( skey, obs_full, train_size, test_size, batch_size, train_batches, test_batches):
    ii = random.permutation(skey, obs_full.shape[-1])
    itrain, itest = ii[:train_size], ii[train_size:train_size+test_size]
    train_idx = itrain.reshape(train_batches, batch_size)
    obs_train = jnp.moveaxis(obs_full[...,train_idx],2,0)
    test_idx = itest.reshape(test_batches, batch_size)
    obs_test = jnp.moveaxis(obs_full[...,test_idx],2,0)
    return obs_train, itrain, obs_test, itest


data_suffix_list = np.array([x[40:-4] for x in glob.glob('mean_im_de_trj/*.mat')])
data_path = 'mean_im_de_trj'

def get_spiketrains(filename,threshold = 5):
    x=io.loadmat(filename)
    data = x['spktrains_mat']
    ins = x['envinputs_mat']
    sums = [np.sum(q,axis=0) for q in data]
    if len(sums[0].shape) ==0:
        return False, None, None
    zeros = np.array([s == 0 for s in sums])
    real_spike_train = ~zeros
    idx = np.multiply(*real_spike_train)
    sum_idx = np.sum(idx)
    if sum_idx < threshold:
        return False, None, None
    clean_data = data[:,:,idx]
    return True, sum_idx, clean_data

def data_loader():
    recordings=[]
    obs_list=[]
    counts=[]
    for data_suffix in data_suffix_list:
        file_mat = f'spktrains_mean_by_choice_{data_suffix}'
        flag, ncells, obs_ = get_spiketrains(
            os.path.join(data_path, file_mat)        
        )
        if flag:
            obs_list.append( obs_ )
            ncells = obs_.shape[-1]
            counts.append( ncells )
    obs = jnp.concatenate( obs_list, axis=2 )
    counts = np.array( counts )
    return counts, obs
