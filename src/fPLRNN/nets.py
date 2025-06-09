from flax import linen as nn
from jax import numpy as jnp
from . import cells

class PLRNNet( nn.Module ):
    subsets: int
    contexts: int
    length: int
    features: int
    num_neurons: int    
    @nn.compact
    def __call__(self):
        length_minus = self.length-1
        RNN = nn.scan(
            cells.PLRNNCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            out_axes = -2,
            length = length_minus
        )
        latent_model = RNN(name='latent_model',features = self.features)
        vectorizedDense = nn.vmap(
            nn.Dense,
            in_axes = 0, out_axes = 0,
            variable_axes={'params':0},
            split_rngs={'params':True}
        )
        observation_model= vectorizedDense(
            self.num_neurons,
            use_bias=False,
            name = 'observation_model'
        )
        z = jnp.zeros(
            (self.subsets, self.contexts, self.length , self.features))
        z0 = self.param(
            'z0',
            nn.initializers.normal(),
            ( self.subsets, self.contexts, self.features )
        )
        z = z.at[ ..., 0 , : ].set( z0 )
        z_last , z_length_minus = latent_model(z0,None)            
        z = z.at[ ..., 1:, : ].set( z_length_minus )
        x = observation_model(z)
        return x,z

class LRNNet( nn.Module ):
    subsets: int
    contexts: int
    length: int
    features: int
    num_neurons: int    
    @nn.compact
    def __call__(self):
        length_minus = self.length-1
        RNN = nn.scan(
            cells.LRNNCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            out_axes = -2,
            length = length_minus
        )
        latent_model = RNN(name='latent_model',features = self.features)
        vectorizedDense = nn.vmap(
            nn.Dense,
            in_axes = 0, out_axes = 0,
            variable_axes={'params':0},
            split_rngs={'params':True}
        )
        observation_model= vectorizedDense(
            self.num_neurons,
            use_bias=False,
            name = 'observation_model'
        )
        z = jnp.zeros(
            (self.subsets, self.contexts, self.length , self.features))
        z0 = self.param(
            'z0',
            nn.initializers.normal(),
            ( self.subsets, self.contexts, self.features )
        )
        z = z.at[ ..., 0 , : ].set( z0 )
        z_last , z_length_minus = latent_model(z0,None)            
        z = z.at[ ..., 1:, : ].set( z_length_minus )
        x = observation_model(z)
        return x,z
    
