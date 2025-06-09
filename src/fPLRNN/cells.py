from flax import linen as nn
from functools import partial

class LRNNCell(nn.RNNCellBase):
    features: int
    @nn.compact
    def __call__(self, z, _):
        A = self.param(
            'A',
            nn.initializers.zeros_init(), 
            (self.features,),
        )
        dense_h = partial(
            nn.Dense,
            features = self.features,
            kernel_init = nn.initializers.zeros_init(),
            bias_init = nn.initializers.normal(),
        )
        Wh_terms = dense_h(name = 'Wh')( z )
        A_term = z*A
        new_z = A_term + Wh_terms
        return new_z,new_z


class PLRNNCell(nn.RNNCellBase):
    features: int
    @nn.compact
    def __call__(self, z, _):
        A = self.param(
            'A',
            nn.initializers.zeros_init(), 
            (self.features,),
        )
        dense_h = partial(
            nn.Dense,
            features = self.features,
            kernel_init = nn.initializers.zeros_init(),
            bias_init = nn.initializers.normal(),
        )
        Wh_terms = dense_h(name = 'Wh')( nn.relu(z) )
        A_term = z*A
        new_z = A_term + Wh_terms
        return new_z,new_z

