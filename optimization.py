import optax
import jax
from jax import numpy as jnp
from jax import random, lax, vmap,value_and_grad, tree_util
import numpy as np
from functools import partial

get_CCs = vmap( vmap(
    vmap( lambda a,b: jnp.corrcoef(a,b)[0,1], in_axes = (1,1) )
) )

def init_AW(key,D,scale=10):
    R = random.normal(key,(D,D))
    q = (R.T@R/D  + scale*jnp.eye(D) ) 
    w,v = jnp.linalg.eigh(q)
    H = q / jnp.max(w.real)
    A = jnp.diag(H)
    W = H - A*jnp.eye(D)
    return A,W

def scramble(key,mm):
    off_diag =  ~jnp.eye( mm.shape[0],dtype=bool )
    rr = random.permutation( key, mm[off_diag] )
    mm = mm.at[off_diag].set( rr )
    return mm

def shuffle( T, interval_size ):
    tt=np.arange(T)
    C = int(T/interval_size)
    ttC = np.array_split(tt,C)
    Cr = np.random.permutation( C )
    return np.concatenate( [ttC[c] for c in Cr] )

def computeLossRNN(params,net,obs,hyperparameters):
    l1_ratio_B, L_Breg = hyperparameters
    x,z = net.apply(params)
    B = params['params']['observation_model']['kernel']
    catB = jnp.hstack(B)
    # stack B matrices each from different batches into one matrix
    # so the matrix "catB" is the "full" B matrix
    LB1 = jnp.linalg.norm( catB, ord=1 ) 
    LB2 = jnp.linalg.norm( catB )**2 
    alpha = l1_ratio_B 
    LB = alpha*LB1 + (1-alpha)*LB2
    error = obs - x
    MAE = jnp.mean( jnp.abs( error))
    loss = MAE + L_Breg*LB
    return loss , MAE 

def optLoopRNN_(f_state,_,loss_grad,optimizer):
    params , opt_state = f_state
    (loss,MAE),grads = loss_grad(params)
    updates, opt_state = optimizer.update(grads, opt_state,params)
    params = optax.apply_updates(params, updates)
    params['params']['latent_model']['Wh']['kernel'] = jnp.fill_diagonal(params['params']['latent_model']['Wh']['kernel'],0,wrap=False,inplace=False)
    return (params, opt_state), (loss, MAE)

def computeLossObs(params,net,z,obs,hyperparameters):
    l1_ratio_B, L_Breg = hyperparameters
    x = net.apply(params,z)
    B = params['params']['kernel']
    catB = jnp.hstack(B)
    LB1 = jnp.linalg.norm( catB, ord=1 ) 
    LB2 = jnp.linalg.norm( catB )**2 
    alpha = l1_ratio_B 
    LB = alpha*LB1 + (1-alpha)*LB2 
    error = obs - x
    MAE = jnp.mean( jnp.abs( error))
    loss = MAE + L_Breg*LB
    return loss , MAE

def optLoopObs_(f_state,_,loss_grad,optimizer):
    params , opt_state = f_state
    (loss,MAE),grads = loss_grad(params)
    updates, opt_state = optimizer.update(grads, opt_state,params)
    params = optax.apply_updates(params, updates)
    return (params, opt_state), (loss, MAE)

def runOptimizationObs(net,params,optimizer,loss_function,epochs,obs,z,hyperparameters):
    opt_state = optimizer.init(params)
    compute_loss = partial(
        loss_function,
        net = net,
        z = z,
        obs = obs,
        hyperparameters = hyperparameters
    )
    loss_grad = value_and_grad(compute_loss,has_aux=True)
    loss = compute_loss( params )
    optLoopObs = partial( optLoopObs_, loss_grad = loss_grad,optimizer=optimizer )
    (params,opt_state),(losses,MAEs) = lax.scan(optLoopObs,(params,opt_state),xs=None,length=epochs)
    x = net.apply(params,z)
    return x,params ,losses, MAEs

def runOptimizationRNN(net,params,optimizer,loss_function,epochs,obs,hyperparameters):
    opt_state = optimizer.init(params)    
    compute_loss = partial(
        loss_function,
        net = net,
        obs = obs,
        hyperparameters = hyperparameters
    )
    loss_grad = value_and_grad(compute_loss,has_aux=True)
    loss = compute_loss( params )
    optLoopRNN = partial( optLoopRNN_, loss_grad = loss_grad,optimizer=optimizer )
    (params,opt_state),(losses,MAEs) = lax.scan(optLoopRNN,(params,opt_state),xs=None,length=epochs)
    x,z = net.apply(params)
    return z,x,params ,losses, MAEs

def modelInstanceRNN(
        key,
        obs_train, obs_test,
        Net,        
        dz,
        epochs,
        hyperparameters,
):
    S1,S2 = epochs
    train_batches = obs_train.shape[0]
    os = obs_test.shape
    test_batches = os[0]
    batch_size = os[3]
    T = os[2]
    ## training
    optimizer_train = optax.chain(
        optax.clip( 0.2 ),
        optax.multi_transform(
            { 'adam':optax.adam(learning_rate=1e-3),
              'set0':optax.set_to_zero() },
            { 'params':
              {'observation_model':'adam',
               'z0':'adam',
               'latent_model':{'A':'adam','Wh':'adam'} } }
        )
    )
    

    key,skey = random.split(key)

    
    net = Net( train_batches,2,T,dz,batch_size )
    params = net.init(skey)
    print( jax.tree.map( lambda a:a.shape, params))
    key,skey = random.split(key)
    A_term, W_term = init_AW( skey, dz )

    params['params']['latent_model']['A'] = A_term
    params['params']['latent_model']['Wh']['kernel'] = W_term

    z_training,x_training,params,losses_train, MAEs_train = runOptimizationRNN(
        net, params,
        optimizer_train,
        computeLossRNN,
        S1,
        obs_train,
        hyperparameters
    )


    ## testing; SW control
    
    optimizer_transfer_learning = optax.chain(
        optax.clip( 0.2 ),
        optax.multi_transform(
            { 'adam':optax.adam(learning_rate=1e-3),
              'set0':optax.set_to_zero() },
            {'params':
             { 'observation_model':'adam',
               'z0':'adam',
               'latent_model':'set0' } }
        )  
    )
    

    # make dict TL
    key,skey = random.split(key)
    net = Net( test_batches,2,T,dz,batch_size )
    params_TL = net.init( skey ) # TL <- Transfer Learning

    
    params_TL['params']['latent_model'] = tree_util.tree_map(
        lambda x:x,
        params['params']['latent_model']
    )

    # make dict SW
    key,skey = random.split(key)
    params_SW = net.init( skey ) # SW <- Scramble W


    params_SW['params']['latent_model'] = tree_util.tree_map(
        lambda x:x,
        params['params']['latent_model']
    )
    W = params['params']['latent_model']['Wh']['kernel']    
    key,skey = random.split(key)
    SW = scramble( skey, W )
    params_SW['params']['latent_model']['Wh']['kernel'] = SW

    vro = vmap( runOptimizationRNN, in_axes = (None,0,None,None,None,None,None))
    (z_testing,z_SW),(x_testing,x_SW),params_TL_SW,(losses_TL,losses_SW), (MAEs_TL,MAEs_SW)=vro(
        net,
        tree_util.tree_map(
            lambda x,y:jnp.stack((x,y)),
            params_TL, params_SW
        ),
        optimizer_transfer_learning,
        computeLossRNN,
        S2,
        obs_test,
        hyperparameters
    )
    params_TL = tree_util.tree_map( lambda x:x[0], params_TL_SW )
    params_SW = tree_util.tree_map( lambda x:x[1], params_TL_SW )

    
    CCs_train = get_CCs( x_training, obs_train )
    CCs_test =  get_CCs( x_testing, obs_test )
    CCs_SW =    get_CCs( x_SW, obs_test )
    
    train_result = ( x_training, z_training, losses_train, MAEs_train, CCs_train, params )
    test_result =  ( x_testing,  z_testing,  losses_TL,    MAEs_TL,    CCs_test, params_TL )
    SW_result =    ( x_SW,       z_SW,       losses_SW,    MAEs_SW,    CCs_SW,   params_SW )

    return train_result, test_result, SW_result

def modelInstanceObs(
        key, 
        obs_test,
        z_test,
        net,
        dz,
        epochs,
        sh_idx,
        hyperparameters,
):
    os = obs_test.shape
    test_batches = os[0]
    batch_size = os[3]
    T = os[2]
    _,S2 = epochs
    optimizer = optax.chain(
        optax.clip( 0.2 ),
        optax.multi_transform(
            { 'adam':optax.adam(learning_rate=1e-3) },
            { 'params':'adam' }
        )
    )
    z_sh = z_test[...,sh_idx,:]
    key,skey = random.split(key)
    params = net.init( skey, z_sh )
    key,skey = random.split(key)
    x,params,losses,MAEs = runOptimizationObs(
        net,params,optimizer,computeLossObs,S2,obs_test,z_sh,hyperparameters )
    CCs = get_CCs( x, obs_test )
    sh_result = ( x, z_sh, losses, MAEs, CCs, params )
    return sh_result
