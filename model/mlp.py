import mxnet as mx
from mxnet.initializer import Xavier, Zero
import numpy as np
from utils.mxnet_util import _positive_transform

class MLP_encoder1_cartesian(mx.gluon.HybridBlock):
    def __init__(self, units, act='tanh', dtype=np.float32, prefix=None, **kwargs): #'tanh'
        super(MLP_encoder1_cartesian, self).__init__(prefix=prefix, **kwargs)
        self.dim_latent = units[-1]
        self.units = units[:-1] + [2*units[-1]]
        self.nLayers = len(units)-1
        self.dtype=dtype
        with self.name_scope():
            for i in range(self.nLayers):
                setattr(self,'dense_'+str(i), mx.gluon.nn.Dense(self.units[i+1], activation=act if i<self.nLayers-1 else None,
                           weight_initializer=Xavier(magnitude=2.),flatten=False, dtype=self.dtype))

    def hybrid_forward(self, F, x):
        for i in range(self.nLayers):
            x = getattr(self, 'dense_'+str(i))(x)
        units = F.split(x, axis=1, num_outputs=2)
        return units[0], _positive_transform(F, units[1])

    def push(self, F, x):
        return x

class MLP_encoder2_cartesian(mx.gluon.HybridBlock):
    def __init__(self, prefix=None, **kwargs): #'tanh'
        super(MLP_encoder2_cartesian, self).__init__(prefix=prefix, **kwargs)

    def hybrid_forward(self, F, x):
        return x, None
    
class MLP_decoder(mx.gluon.HybridBlock):
    def __init__(self, units, act='tanh', binary=True, dtype=np.float32, prefix=None, **kwargs): #'tanh'
        super(MLP_decoder, self).__init__(prefix=prefix, **kwargs)
        self.units = units
        self.nLayers = len(units)-1
        self.dtype=dtype
        self.binary = binary
        with self.name_scope():
            for i in range(self.nLayers):
                setattr(self,'dense_'+str(i), mx.gluon.nn.Dense(units[i+1], activation=act if i<self.nLayers-1 else None,
                           weight_initializer=Xavier(magnitude=2.),flatten=False, dtype=self.dtype))

    def hybrid_forward(self, F, x):
        for i in range(self.nLayers):
            x = getattr(self, 'dense_'+str(i))(x)
        if self.binary:
            return F.sigmoid(x)
        else:
            return x
    
    def push(self, F, x):
        return x

class MLP_decoder_mixed_data(mx.gluon.HybridBlock):
    def __init__(self, units, act='tanh', dtype=np.float32, prefix=None, binary=[], categorical=[], continuous=[], **kwargs): #'tanh'
        super(MLP_decoder_mixed_data, self).__init__(prefix=prefix, **kwargs)
        self.units = units
        self.nLayers = len(units) - 1
        self.dtype = dtype    
    
        self.num_variables = units[-1]
    
    
        tmp = np.zeros(self.num_variables)
        
        binary_ = tmp.copy()
        binary_[binary] = 1
        self.binary_ = binary
        self.binary = mx.nd.array(binary_)
        
        categorical_ = []
        for cat in categorical:
            if len(cat)>0:
                cat_i = tmp.copy()
                cat_i[cat] = 1
                categorical_ += [mx.nd.array(cat_i)]
        self.categorical_ = categorical
        self.categorical = categorical_
        
        continuous_ = tmp.copy()
        continuous_[continuous] = 1
        self.continuous = mx.nd.array(continuous_)
        
        
        self.units = units[:-1] + [self.num_variables+len(continuous)]
        
        with self.name_scope():
            for i in range(self.nLayers):
                setattr(self,'dense_'+str(i), mx.gluon.nn.Dense(self.units[i+1], activation=act if i<self.nLayers-1 else None,
                           weight_initializer=Xavier(magnitude=2.),flatten=False, dtype=self.dtype))

    def hybrid_forward(self, F, x, soft_zero = 1e-6):
        for i in range(self.nLayers):
            x = getattr(self, 'dense_'+str(i))(x)
        x_mean = x[:, :self.num_variables]
        
        x_std = None
        if x.shape[1] > self.num_variables:
            x_std = _positive_transform(F, x[:, self.num_variables:])
        
        x_mean_t = F.transpose(x_mean)
        
        x_mean_t =  F.where(self.binary>=1, F.sigmoid(x_mean_t), x_mean_t)
        categorical_variables = []
        for cat_i, cat in enumerate(self.categorical):
            tmp = F.Activation(x_mean_t[self.categorical_[cat_i], :], act_type='softrelu')+soft_zero
            tmp = F.sum(tmp , axis=0, keepdims=True)
            x_mean_t =  F.where(cat>=1, (F.Activation(x_mean_t, act_type='softrelu') + soft_zero) / tmp , x_mean_t)
        #Continuous variables stay as they are
        return F.transpose(x_mean_t), x_std
    
    def push(self, F, x):
        x_mean, x_std = x
        for cat in self.categorical_:
            max_i = np.array(F.argmax(x_mean[:,cat], axis=1).asnumpy(), dtype=int)
            #import pdb; pdb.set_trace()
            x_mean[:, cat] = F.zeros((x_mean.shape[0], len(cat)),dtype=self.dtype) 
            for ind, max_ii in enumerate(max_i):
                x_mean[ind, cat[max_ii]] = F.ones((1,), dtype=self.dtype)
        if len(self.binary_)>0:
            x_mean[:,self.binary_] = F.round(x_mean[:,self.binary_])
        return x_mean, x_std

class MLP(mx.gluon.HybridBlock):
    def __init__(self, units, act='tanh', dtype=np.float32, prefix=None, last_linear=True, **kwargs): #'tanh'
        super(MLP, self).__init__(prefix=prefix, **kwargs)
        self.units = units
        self.nLayers = len(units)-1
        self.dtype=dtype
        with self.name_scope():
            for i in range(self.nLayers):
                activation=act
                if last_linear and i==self.nLayers-1:
                    activation=None
                setattr(self,'dense_'+str(i), mx.gluon.nn.Dense(units[i+1], activation=activation,
                           weight_initializer=Xavier(magnitude=2.),flatten=False, dtype=self.dtype)) 
    
    def hybrid_forward(self, F, x):
        for i in range(self.nLayers):
            x = getattr(self, 'dense_'+str(i))(x)
        return x