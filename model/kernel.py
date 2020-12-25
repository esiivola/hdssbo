import numpy as np
import mxnet as mx

from utils.mxnet_util import _positive_transform, _positive_transform_reverse


def get_R2(F, x, x2=None, lengthscale=1.0):
    if x2 is None:
        xsc = x / lengthscale
        amat = F.linalg.syrk(xsc) * -2
        dg_a = F.sum(F.square(xsc), axis=-1)
        amat = F.broadcast_add(amat, F.expand_dims(dg_a, axis=-1))
        amat = F.broadcast_add(amat, F.expand_dims(dg_a, axis=-2))
    else:
        x1sc = x / lengthscale
        x2sc = x2 / lengthscale
        amat = F.linalg.gemm2(x1sc, x2sc, False, True) * -2
        dg1 = F.sum(F.square(x1sc), axis=-1, keepdims=True)
        amat = F.broadcast_add(amat, dg1)
        dg2 = F.expand_dims(F.sum(F.square(x2sc), axis=-1), axis=-2)
        amat = F.broadcast_add(amat, dg2)
    return amat

class Kernel(object):
    def __init__(self):
        self._block_K = None
        self._block_Kdiag = None

    def K(self, X, X2=None):
        return self._block_K(X, X2)

    def Kdiag(self, X):
        return self._block_Kdiag(X)

    @property
    def params(self):
        return self._params

class Linear(Kernel):
    class Gluon_K(mx.gluon.HybridBlock):
        def __init__(self, input_dim, ARD, dtype, prefix=None, params=None):
            super(Linear.Gluon_K, self).__init__(prefix=prefix, params=params)
            self.input_dim = input_dim
            self.ARD = ARD
            with self.name_scope():
                self.offset = self.params.get('offset')
                self.variances_b = self.params.get('variances_b')
                self.variances = self.params.get('variances')
    
        def hybrid_forward(self, F, x, x2=None, **kwargs):
            variances, offset, variances_b = _positive_transform(F, kwargs['variances']), kwargs['offset'], _positive_transform(F, kwargs['variances_b'])
            #import pdb; pdb.set_trace()
            if self.ARD:
                var_sqrt = F.expand_dims(F.sqrt(variances), axis=-2)
                if x2 is None:
                    xsc = (x - offset) * var_sqrt
                    return variances_b +  F.linalg.syrk(xsc)
                else:
                    xsc = (x - offset) * var_sqrt
                    x2sc = (x2 - offset) * var_sqrt
                    return variances_b + F.linalg.gemm2(xsc, x2sc, False, True)
            else:
                if X2 is None:
                    A = F.linalg.syrk(x - offset)
                else:
                    A = F.linalg.gemm2(x - offset, x2 - offset, False, True)
                return variances_b + A * F.expand_dims(variances, axis=-1)

    class Gluon_Kdiag(mx.gluon.HybridBlock):
        def __init__(self, input_dim, ARD, dtype, prefix=None, params=None):
            super(Linear.Gluon_Kdiag, self).__init__(prefix=prefix, params=params)
            self.input_dim = input_dim
            self.ARD = ARD
            with self.name_scope():
                self.offset = self.params.get('offset')
                self.variances_b = self.params.get('variances_b')
                self.variances = self.params.get('variances')

        def hybrid_forward(self, F, x, **kwargs):
            variances, offset, variances_b = _positive_transform(F, kwargs['variances']), kwargs['offset'], _positive_transform(F, kwargs['variances_b'])
            x2 = F.square(x-offset)
            return variances_b + F.reshape(F.sum(x2 * F.expand_dims(variances, axis=-2), axis=-1), (-1,1))

    def __init__(self, input_dim, ARD=False, variances=1., offset=0., variances_b=0.0001, name='linear',
                 active_dims=None, dtype=np.float64, ctx=None, fix=False):
        self.ARD = ARD
        #variances = Variable(shape=(input_dim if ARD else 1,), initial_value=variances)
        #offset = Variable(shape=(1,input_dim if ARD else 1,1), initial_value=offset)
        #variances_b = Variable(shape=(1,), initial_value=variances_b)
        offset = np.full(input_dim, offset).reshape((1,-1)) if ARD else [offset]
        variances = np.full(input_dim, variances) if ARD else [variances]
        if fix:
            grad_req="null"
        else:
            grad_req="write"
        #import pdb; pdb.set_trace()
        self._params = mx.gluon.ParameterDict(prefix=name+'_')
        self._params.get('variances', shape=(input_dim,) if ARD else (1,), dtype=dtype, allow_deferred_init=True, grad_req=grad_req)
        self._params.get('offset', shape=(1,input_dim) if ARD else (1,1), dtype=dtype, allow_deferred_init=True, grad_req=grad_req)
        self._params.get('variances_b', shape=(1,), dtype=dtype, allow_deferred_init=True, grad_req=grad_req)
        self._params.initialize(ctx=ctx)
        #import pdb; pdb.set_trace()
        self._params.get('variances').set_data(_positive_transform_reverse(mx.nd, mx.nd.array(variances, dtype=dtype, ctx=ctx)))
        self._params.get('offset').set_data( mx.nd.array(offset, dtype=dtype, ctx=ctx))
        self._params.get('variances_b').set_data(_positive_transform_reverse(mx.nd, mx.nd.array([variances_b], dtype=dtype, ctx=ctx)))
        
        self._block_K = self.Gluon_K(input_dim, ARD, dtype, prefix=name+'_K', params=self._params)
        self._block_Kdiag = self.Gluon_Kdiag(input_dim, ARD, dtype, prefix=name+'_Kdiag', params=self._params)


class Exponential(Kernel):
    class Gluon_K(mx.gluon.HybridBlock):
        def __init__(self, input_dim, ARD, dtype, prefix=None, params=None):
            pass

        def hybrid_forward(self, F, x, x2=None, **kwargs):
            pass

    class Gluon_Kdiag(mx.gluon.HybridBlock):
        def __init__(self, input_dim, ARD, dtype, prefix=None, params=None):
            super(Exponential.Gluon_Kdiag, self).__init__(prefix=prefix, params=params)
            self.input_dim = input_dim
            self.ARD = ARD
            with self.name_scope():
                self.lengthscale = self.params.get('lengthscale')
                self.variance= self.params.get('variance')

        def hybrid_forward(self, F, x, **kwargs):
            variance = _positive_transform(F, kwargs['variance'])
            return F.broadcast_mul(F.ones_like(F.slice_axis(x, axis=1, begin=0, end=1)),variance)

    def __init__(self, input_dim, ARD=False, variance=1., lengthscale=1., dtype=np.float64, ctx=mx.cpu(), name='rbf', fix=False):
        self.name=name
        self.dtype = dtype
        self.input_dim = input_dim
        self.ARD = ARD
        lengthscale = np.full(input_dim, lengthscale) if ARD else lengthscale

        if fix:
            grad_req="null"
        else:
            grad_req="write"
        self._params = mx.gluon.ParameterDict(prefix=name+'_')
        self._params.get('lengthscale', shape=(input_dim,) if ARD else (1,), dtype=dtype, allow_deferred_init=True, grad_req=grad_req)
        self._params.get('variance', shape=(1,), dtype=dtype, allow_deferred_init=True, grad_req=grad_req)
        self._params.initialize(ctx=ctx)
        self._params.get('variance').set_data(_positive_transform_reverse(mx.nd, mx.nd.array([variance], dtype=dtype, ctx=ctx)))
        self._params.get('lengthscale').set_data(_positive_transform_reverse(mx.nd, mx.nd.array(lengthscale,dtype=dtype, ctx=ctx)))

        self._block_K = self.Gluon_K(input_dim, ARD, dtype, prefix=name+'_K', params=self._params)
        self._block_Kdiag = self.Gluon_Kdiag(input_dim, ARD, dtype, prefix=name+'_Kdiag', params=self._params)

class RBF(Exponential):
    class Gluon_K(mx.gluon.HybridBlock):
        def __init__(self, input_dim, ARD, dtype, prefix=None, params=None):
            super(RBF.Gluon_K, self).__init__(prefix=prefix, params=params)
            self.input_dim = input_dim
            self.ARD = ARD
            with self.name_scope():
                self.lengthscale = self.params.get('lengthscale')
                self.variance= self.params.get('variance')
    
        def hybrid_forward(self, F, x, x2=None, **kwargs):
            lengthscale, variance = _positive_transform(F, kwargs['lengthscale']), _positive_transform(F, kwargs['variance'])
            lengthscale = F.expand_dims(lengthscale, axis=-2)
            R2 = get_R2(F,x,x2=x2,lengthscale=lengthscale)            
            return F.exp(R2 / -2) * F.expand_dims(variance, axis=-1)

class Matern52(Exponential):
    class Gluon_K(mx.gluon.HybridBlock):
        def __init__(self, input_dim, ARD, dtype, prefix=None, params=None):
            super(Matern52.Gluon_K, self).__init__(prefix=prefix, params=params)
            self.input_dim = input_dim
            self.ARD = ARD
            with self.name_scope():
                self.lengthscale = self.params.get('lengthscale')
                self.variance= self.params.get('variance')
    
        def hybrid_forward(self, F, x, x2=None, **kwargs):
            lengthscale, variance = _positive_transform(F, kwargs['lengthscale']), _positive_transform(F, kwargs['variance'])
            lengthscale = F.expand_dims(lengthscale, axis=-2)
            R2 = get_R2(F,x,x2=x2,lengthscale=lengthscale)
            R = F.sqrt(F.clip(R2, 1e-14, np.inf))
            return F.broadcast_mul(
                (1+np.sqrt(5)*R+5/3.*R2)*F.exp(-np.sqrt(5)*R),
                F.expand_dims(variance, axis=-2))

class Matern32(Exponential):
    class Gluon_K(mx.gluon.HybridBlock):
        def __init__(self, input_dim, ARD, dtype, prefix=None, params=None):
            super(Matern32.Gluon_K, self).__init__(prefix=prefix, params=params)
            self.input_dim = input_dim
            self.ARD = ARD
            with self.name_scope():
                self.lengthscale = self.params.get('lengthscale')
                self.variance= self.params.get('variance')
    
        def hybrid_forward(self, F, x, x2=None, **kwargs):
            lengthscale, variance = _positive_transform(F, kwargs['lengthscale']), _positive_transform(F, kwargs['variance'])
            lengthscale = F.expand_dims(lengthscale, axis=-2)
            R2 = get_R2(F,x,x2=x2,lengthscale=lengthscale)
            R = F.sqrt(F.clip(R2, 1e-14, np.inf))
            return F.broadcast_mul(
                (1+np.sqrt(3)*R)*F.exp(-np.sqrt(3)*R),
                F.expand_dims(variance, axis=-2))

class Add(object):
    class Gluon_K(mx.gluon.HybridBlock):
        def __init__(self, Klist, prefix=None, params=None):
            super(Add.Gluon_K, self).__init__(prefix=prefix, params=params)
            self.Klist = Klist
            for k in Klist:
                self.register_child(k._block_K)

        def hybrid_forward(self, F, x, x2=None, **kwargs):
            K = self.Klist[0]._block_K(x,x2)
            for k in self.Klist[1:]:
                K = K+k._block_K(x,x2)
            return K

    class Gluon_Kdiag(mx.gluon.HybridBlock):
        def __init__(self, Klist, prefix=None, params=None):
            super(Add.Gluon_Kdiag, self).__init__(prefix=prefix, params=params)
            self.Klist = Klist
            for k in Klist:
                self.register_child(k._block_Kdiag)

        def hybrid_forward(self, F, x, **kwargs):
            K = self.Klist[0]._block_Kdiag(x)
            for k in self.Klist[1:]:
                K = K+k._block_Kdiag(x)
            return K

    def __init__(self, Klist, name='add'):
        self.name=name
        self.Klist = Klist

        self._block_K = self.Gluon_K(Klist, prefix=name+'_K', params=self.params)
        self._block_Kdiag = self.Gluon_Kdiag(Klist, prefix=name+'_Kdiag', params=self.params)

    @property
    def params(self):
        ret = mx.gluon.ParameterDict(prefix=self.name+'_')
        for k in self.Klist:
            ret.update(k.params)
        return ret

class Multiply(object):
    class Gluon_K(mx.gluon.HybridBlock):
        def __init__(self, Klist, prefix=None, params=None):
            super(Multiply.Gluon_K, self).__init__(prefix=prefix, params=params)
            self.Klist = Klist
            for k in Klist:
                self.register_child(k._block_K)

        def hybrid_forward(self, F, x, x2=None, **kwargs):
            K = self.Klist[0]._block_K(x,x2)
            for k in self.Klist[1:]:
                K = K*k._block_K(x,x2)
            return K

    class Gluon_Kdiag(mx.gluon.HybridBlock):
        def __init__(self, Klist, prefix=None, params=None):
            super(Multiply.Gluon_Kdiag, self).__init__(prefix=prefix, params=params)
            self.Klist = Klist
            for k in Klist:
                self.register_child(k._block_Kdiag)

        def hybrid_forward(self, F, x, **kwargs):
            K = self.Klist[0]._block_Kdiag(x)
            for k in self.Klist[1:]:
                K = K*k._block_Kdiag(x)
            return K

    def __init__(self, Klist, name='add'):
        self.name=name
        self.Klist = Klist

        self._block_K = self.Gluon_K(Klist, prefix=name+'_K', params=self.params)
        self._block_Kdiag = self.Gluon_Kdiag(Klist, prefix=name+'_Kdiag', params=self.params)

    @property
    def params(self):
        ret = mx.gluon.ParameterDict(prefix=self.name+'_')
        for k in self.Klist:
            ret.update(k.params)
        return ret
