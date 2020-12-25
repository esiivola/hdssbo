import numpy as np
import mxnet as mx
from utils.mxnet_util import _positive_transform, make_diagonal, _positive_transform_reverse
from GPy.inference.latent_function_inference.posterior import Posterior
from GPy.kern import RBF
from model import mlp
from model import costs

def compute_conditional_f(F, L, Kuf, Kff, mu, Ls=None, diag=False):
    # Compute marginal distributions at requested points
    Kmmi = F.linalg.potri(L) #Kmm^{-1}
    #import pdb; pdb.set_trace()
    KmmiKuf = F.linalg.gemm2(Kmmi, Kuf) # Kmm^{-1} Kmn = A^{\Top}
    mu_f = F.linalg.gemm2(mu, KmmiKuf) # mu_f = A mu (N x C)
    if not diag:
        aux = Kff - F.linalg.gemm2(Kuf, KmmiKuf, transpose_a=True)
        aux = mx.nd.expand_dims(aux,0)
        v_f = aux
        if Ls is not None:
            KmmiKufbrod = F.expand_dims(KmmiKuf, axis=0)
            tmp = F.linalg.gemm2(Ls, KmmiKufbrod, transpose_a=True) # Ls^\Top A^\Top
            v_f = aux + F.linalg.gemm2(tmp, tmp, transpose_a=True) #F.sum(F.square(tmp),0) = sum((Ls^\Top A^\Top) * (Ls^\Top A^\Top), 0) = diag(A S A^\Top) (as a row)
    else:
        v_f = (F.expand_dims(mx.nd.diag(Kff), axis=1) - F.expand_dims(F.sum(KmmiKuf*Kuf,0), axis=1)).T # diag(Knn) - diag(Knm Kmm^{-1} Kmn) (as a row)
        if Ls is not None:
            KmmiKufbrod = F.expand_dims(KmmiKuf, axis=0)
            aux_shape = list(KmmiKufbrod.shape)
            aux_shape[-3] = Ls.shape[-3]
            KmmiKufbrod = KmmiKufbrod.broadcast_to(tuple(aux_shape))
            tmp = F.linalg.gemm2(Ls, KmmiKufbrod, transpose_a=True) # Ls^\Top A^\Top
            v_f = v_f + F.sum(F.square(tmp),-2) #F.sum(F.square(tmp),0) = sum((Ls^\Top A^\Top) * (Ls^\Top A^\Top), 0) = diag(A S A^\Top) (as a row)
    return mu_f, v_f

class InitCaller(type):
    def __call__(cls, *args, **kwargs):
        """Called when you call MyNewClass() """
        obj = type.__call__(cls, *args, **kwargs)
        obj.post_init()
        return obj

class ObservationModel(mx.gluon.HybridBlock, metaclass=InitCaller):
    def __init__(self):
        raise NotImplementedError

    def post_init(self):
        pass

    def predict(self, F, z_p):
        raise NotImplementedError

    def hybrid_forward(self, F, z_p, y_p):
        raise NotImplementedError

    def freeze_params(self):
        raise NotImplementedError

    def data_changed(self, X, Y, YC):
        raise NotImplementedError

    def get_labeled_ind(self, Y):
        raise NotImplementedError

    def get_batch_from_inds(self, X, Y, inds, W=None):
        raise NotImplementedError
    def set_inducing_to_data(self, X_latent, Y=None):
        pass

class GP(ObservationModel):
    def __init__(self, kern, ctx=mx.cpu(), dtype=np.float64, params=None, prefix='gp_', fix_sigma=None):
        if params is None:
            params = mx.gluon.ParameterDict(prefix=prefix)
        if fix_sigma is not None:
            constant = fix_sigma
            grad_req="null"
        else:
            constant=0.1
            grad_req="write"
        params.get('sigma2', shape=(1,1), dtype=np.float64,
               init=mx.init.Constant(constant), allow_deferred_init=True, grad_req=grad_req) # +mx.nd.random.normal(shape=(1,1), ctx=ctx, dtype=dtype) 
        params.initialize()
        params.get('sigma2').set_data(_positive_transform_reverse(mx.nd, mx.nd.array([[constant]], dtype=dtype, ctx=ctx)))
        super(ObservationModel, self).__init__(prefix=prefix, params=params)
        self.log_likelihood_y = self._gaussian_log_likelihood
        self.fix_sigma = fix_sigma
        self.dtype = dtype
        self.ctx=ctx
        self.kern = kern
        self.diag_noise = 1e-6

    def post_init(self):
        with self.name_scope():
            self.K = self.kern._block_K
            self.Kdiag = self.kern._block_Kdiag
            self.sigma2 = self.params.get('sigma2')
            self.diag = False

    def get_posterior_parameters(self, F, z_p, z_l=None, y_l=None, **kwargs):
        if z_l is None:
            return F.linalg.potrf(mx.nd.ones((1,1), dtype=self.dtype)), mx.nd.zeros((1,z_p.shape[0]), dtype=self.dtype), self.K(z_p), mx.nd.zeros((1,1), dtype=self.dtype), None
        sigma2 = _positive_transform(F, kwargs['sigma2'])
        Kuu = self.K(z_l)
        Kuf = self.K(z_l, z_p)
        if (kwargs['diag'] if 'diag' in kwargs else self.diag):
            Kff = self.Kdiag(z_p)
        else:
            Kff = self.K(z_p)
        L = F.linalg.potrf(Kuu + F.diag((sigma2.reshape(1) )*mx.nd.ones(z_l.shape[0], dtype=self.dtype)) ) #+ self.diag_noise
        return L, Kuf, Kff, y_l, None

    def _predict(self, F, z_p, z_l=None, y_l=None, **kwargs):
        L, Kuf, Kff, mu, Ls = self.get_posterior_parameters(F, z_p, z_l=z_l, y_l=y_l, **kwargs)
        return compute_conditional_f(F, L, Kuf, Kff, mu, Ls=Ls, diag= kwargs['diag'] if 'diag' in kwargs else self.diag)

    def predict(self, z_p, z_l=None, y_l=None, diag=False):
        kwargs = {'sigma2':self.sigma2.data(), 'diag':diag}
        return self._predict(mx.nd, z_p, z_l, y_l, **kwargs)

    def _gaussian_log_likelihood(self, F, m, v, y, **kwargs):
        sigma2 = _positive_transform(F, kwargs['sigma2'])
        y = mx.nd.array(y, dtype=self.dtype)
        Sinv = 1/ (F.diag(v) + F.diag(sigma2.reshape(1)*mx.nd.ones(v.shape[0], dtype=self.dtype))).reshape(-1)
        ym = F.expand_dipms(y-m, axis=0).reshape(-1)
        
        # COMPUTE FOR EACH POINT, NOT FOR THE JOINT DISTRIBUTION!
        logLik_y = F.sum( -1.0*F.sum(F.log(F.sqrt(Sinv))) - 0.5*v.shape[0]*np.log(2*np.pi) -0.5*F.sum(ym*ym*Sinv)  )
        return logLik_y
        

    def hybrid_forward(self, F, z_p, y_p, z_l, y_l, noiseless, **kwargs):
        m, v = self._predict(F, z_p, z_l, y_l, **kwargs)
        return self.log_likelihood_y(F, m, v, y_p, **kwargs)

    @classmethod
    def get_labelled_ind(cls, Y):
        return np.where(~ np.isnan(Y).flatten())[0]

    def get_labelled(self, X, Y, passed=False):
        labeled = self.get_labelled_ind(Y)
        X_labeled, Y_labeled = None, None
        if np.sum(~ np.isnan(Y).flatten()) > 0:
            X_labeled = X[labeled,:]
            Y_labeled = mx.nd.array(Y[labeled,:], dtype=self.dtype).T
        return X_labeled, Y_labeled
    
    def get_unlabelled(self,X, Y):
        unlabeled = np.where(np.isnan(Y).flatten())[0]
        X_unlabeled = None
        if np.sum(np.isnan(Y).flatten()) > 0:
            X_unlabeled = X[unlabeled,:]
        return X_unlabeled       
        
    @classmethod
    def get_batch_from_inds(cls, X, Y, inds):
        return X[inds,:], Y[inds,:]

    def collect_params(self): #, inducing=False):
        ret = mx.gluon.ParameterDict(self.prefix)
        #if inducing:
        ret.update(self.params)
        ret.update(self.kern.params)
        return ret

    def _set_gp_noise(self, noise):
        if noise is not None:
            self.params.get('sigma2').set_data(mx.nd.array([noise]).reshape((1,1)))
            self.params.get('sigma2').grad_req = 'null'
    

    def update_after_fail(self):
        self.params.get('sigma2').set_data(self.params.get('sigma2').data()*2 )
        #self.kern.params.get('variance').set_data(self.kern.params.get('variance').data() + 0.1)

    def update_sensitive_parameters(self):
        
        lengthscales = _positive_transform(mx.nd, self.kern.params.get('lengthscale').data()).asnumpy().reshape(-1)
        lengthscales = np.clip(lengthscales, 0.3, 5.0)
        lengthscales_ = _positive_transform_reverse(mx.nd, mx.nd.array(lengthscales, dtype=self.dtype))
        self.kern.params.get('lengthscale').set_data(lengthscales_)
        
        rbf_variance = _positive_transform(mx.nd, self.kern.params.get('variance').data()).asnumpy().reshape(-1)
        rbf_variance = np.clip(rbf_variance, 0.5, 5.0)
        rbf_variance_ = _positive_transform_reverse(mx.nd, mx.nd.array(rbf_variance, dtype=self.dtype))
        self.kern.params.get('variance').set_data(rbf_variance_)

        sigma = _positive_transform(mx.nd, self.params.get('sigma2').data()).asnumpy().reshape(-1)
        sigma = np.clip(sigma, 1e-6, 0.1)
        sigma_ = _positive_transform_reverse(mx.nd, mx.nd.array(sigma, dtype=self.dtype))
        self.params.get('sigma2').set_data(sigma_.reshape((1,1)))
        return
        
    def data_changed(self, X, Y, YC):
        pass

class SVGP(GP):
    def __init__(self, kern, dim_h, N_data, ctx=mx.cpu(), dtype=np.float64, params=None, prefix='svgp_', fix_sigma=None, num_inducing=50):
        self.dim_h = dim_h
        self.N = N_data
        self.num_inducing = num_inducing
        if params is None:
            params = mx.gluon.ParameterDict(prefix=prefix)
        params.get('inducing_inputs', shape=(num_inducing, self.dim_h), dtype=dtype,
                   init=mx.init.Constant(mx.nd.random.normal(shape=(num_inducing, self.dim_h), ctx=ctx, dtype=dtype) ), allow_deferred_init=True)#, grad_req='null')
        params.get('qU_mean', shape=(1, num_inducing), dtype=dtype,
                   init=mx.init.Constant(0.0), allow_deferred_init=True)#, grad_req='null')
        params.get('qU_cov_W', shape=(1, num_inducing, num_inducing), dtype=dtype,
                   init=mx.init.Zero(), allow_deferred_init=True)#, grad_req='null')
        params.get('qU_cov_diag', shape=(1, num_inducing), dtype=dtype,
                   init=mx.init.Constant(-5), allow_deferred_init=True)
        
        #Build a mask:
        M = np.tril(np.ones(num_inducing), k=-1)
        self.M = mx.nd.expand_dims(mx.nd.array(M, dtype=dtype), axis=0)
        super(SVGP, self).__init__(kern, ctx=ctx, dtype=dtype, params=params, prefix=prefix, fix_sigma=fix_sigma)
        self.log_likelihood_y = self._gaussian_log_likelihood
        self.initialized_at_random = True
        
    
    def get_minimum(self):
        return np.min(self.params.get('qU_mean').data().asnumpy())
    
    def _gaussian_log_likelihood(self, F, y, L, Kuf, Kff, y_f, Ls, **kwargs): # m, v, y, **kwargs):
        sigma2 = _positive_transform(F, kwargs['sigma2']).reshape(-1)
        lengthscales = _positive_transform(F, kwargs['lengthscales']).reshape(-1)
        sigma_rbf = _positive_transform(F, kwargs['rbf_variance']).reshape(-1)
        y_f = F.array(y_f, dtype=self.dtype).reshape((1,-1))
        Kmmi = F.linalg.potri(L) #Kmm^{-1}
        KmmiKuf = F.linalg.gemm2(Kmmi, Kuf) # Kmm^{-1} Kmn = A^{\Top}
        mu_f = F.linalg.gemm2(y_f, KmmiKuf) # mu_f = A mu (N x C)
        LsTKmmiKuf = F.linalg.gemm2(Ls, F.expand_dims(KmmiKuf, axis=0), transpose_a = True)
        
        y = y.reshape(mu_f.shape)
        ym = F.expand_dims(y-mu_f, axis=0).reshape(-1)
        logLik_y = F.sum( - F.log(F.sqrt(sigma2) * np.sqrt(2*np.pi)) - 0.5*( F.square(ym))/sigma2 )
        
        trace =  -0.5/sigma2 *  F.sum(F.diag( F.linalg.gemm2(LsTKmmiKuf,LsTKmmiKuf, transpose_a = True) ))
        k_tilde = -0.5/sigma2 * F.sum(F.diag( Kff - F.linalg.gemm2(KmmiKuf, Kuf, transpose_a=True)  )) # F.diag(v).reshape(-1)
        
        logLik_prior = F.sum(F.log(lengthscales))*ym.shape[0] + F.sum(F.log(sigma_rbf))*ym.shape[0] 
        return logLik_y + trace + k_tilde + logLik_prior
    
    
    def post_init(self):
        self.params.initialize()
        with self.name_scope():
            self.K = self.kern._block_K
            self.Kdiag = self.kern._block_Kdiag
            self.sigma2 = self.params.get('sigma2')
            self.lengthscales = self.kern.params.get('lengthscale')
            self.rbf_variance = self.kern.params.get('variance')
            self.Z = self.params.get('inducing_inputs')
            self.mu = self.params.get('qU_mean')
            self.S_W = self.params.get('qU_cov_W')
            self.S_diag = self.params.get('qU_cov_diag')
            self.diag = False
            self.eps = 1e-7
    
    def get_GPy_kern(self):
        lengthscale = _positive_transform(mx.nd, self.kern.params.get('lengthscale').data()).asnumpy()
        variance = _positive_transform(mx.nd, self.kern.params.get('variance').data()).asnumpy()
        return RBF(self.dim_h, variance=variance, lengthscale=lengthscale, ARD=True)

    def set_inducing_to_data(self, X_latent, Y=None):
        N_latent = X_latent.shape[0]
        Z = self.params.get('inducing_inputs').data().asnumpy()
        mu = self.params.get('qU_mean').data().asnumpy().reshape((-1,1))
        valid = np.where(False == np.isnan(Z[:,0] ))[0]
        N_valid = len(valid) 
        if Y is not None:
            Y = Y.reshape((-1,1))
        
        Z_new = Z
        mu_new = mu
        
        if (N_latent > N_valid) and (N_valid < self.num_inducing): #Case: there is new data and the total number of data is smaller than the number of inducing points
            Z_new = X_latent # np.r_[Z[:N_valid,:], X_latent[N_valid:,:]]
            if Y is not None:
                mu_new = Y.reshape((-1,1)) #np.r_[mu[:N_valid,:], Y[N_valid:,:].reshape((-1,1))]   
            
        elif self.initialized_at_random: #Case: set_inducing_to_data has not been called before
            self.initialized_at_random = False
            Z_new = X_latent
            if Y is not None:
                mu_new = Y.reshape((-1,1))   
        
        else: #Case: There already is enough inducing points assigned with data
            Z_new = X_latent[:self.num_inducing,:] # np.r_[Z[:N_valid,:], X_latent[N_valid:,:]]
            if Y is not None:
                mu_new = Y.reshape((-1,1))[:self.num_inducing,:]
        
        
        #Set new inducing inputs:
        N_new = Z_new.shape[0]
        if N_new > self.num_inducing:
            Z_new = Z_new[:self.num_inducing,:]
        elif N_new < self.num_inducing: 
            Z_new = np.r_[Z_new, np.nan*np.ones(shape=(self.num_inducing-N_new, self.dim_h)) ]
        
        if N_new > 0:
            self.params.get('inducing_inputs').set_data( Z_new)
        
        
        N_new2 = mu_new.shape[0]
        if N_new2 > self.num_inducing:
            mu_new = mu_new[:self.num_inducing,:]
        elif N_new2 < self.num_inducing:
            mu_new = np.r_[mu_new, np.ones(shape=(self.num_inducing-N_new2, 1)) ]
        if N_new2 > 0:
            self.params.get('qU_mean').set_data(mu_new.T)
        
        
    def predict(self, z_p, diag=False):
        kwargs = {'sigma2':self.sigma2.data(), 'mu':self.mu.data(), 'S_W':self.S_W.data(),
         'S_diag':self.S_diag.data(), 'Z': self.Z.data(), 'diag': diag}
        return self._predict(mx.nd, z_p, **kwargs)
    
    @property
    def posterior(self):
        
        Z = self.Z.data()
        valid = np.where(False == np.isnan(Z.asnumpy()[:,0] ))[0]
        Z = Z[valid,:]
        
        
        mu = self.mu.data()
        S_W = self.S_W.data()
        S_diag = self.S_diag.data()
        
        mu, S_diag, Z = mu[:,valid], S_diag[:,valid], Z[valid,:]
        
        Ls = (S_W*self.M)[np.ix_([0],valid,valid)]  + make_diagonal(mx.nd, S_diag)
        
        
        mu = mu.asnumpy().reshape((-1,1))
        Ls = Ls.asnumpy()[0,:,:]
        K = self.K(Z).asnumpy()
        return Posterior(mean=mu, cov=Ls @ Ls.T, K=K)
    
    @property
    def inducing_points(self):
        Z = self.Z.data()
        valid = np.where(False == np.isnan(Z.asnumpy()[:,0] ))[0]
        return Z[valid,:].asnumpy()
        
    @property
    def sigma2s(self):
        return np.array([_positive_transform(mx.nd, self.sigma2.data()).asnumpy()[0][0]])
        
    def get_posterior_parameters(self, F, z_p, **kwargs):
        mu, S_W, S_diag, Z = kwargs['mu'], kwargs['S_W'], _positive_transform(F, kwargs['S_diag']), kwargs['Z']
        
        valid = np.where(False == np.isnan(Z.asnumpy()[:,0] ))[0]
        mu, S_diag, Z = mu[:,valid], S_diag[:,valid], Z[valid,:]
        
        
        
        Kuu = self.K(Z) + self.eps*F.diag(F.ones(Z.shape[0], dtype=self.dtype))
        Kuf = self.K(Z, z_p)
        if (kwargs['diag'] if 'diag' in kwargs else self.diag):
            Kff = self.Kdiag(z_p)
        else:
            Kff = self.K(z_p)
        L = F.linalg.potrf(Kuu)
        
        Ls = (S_W*self.M)[np.ix_([0],valid,valid)]  + make_diagonal(F, S_diag)#  .reshape(-1))
        return L, Kuf, Kff, mu, Ls
    
    def check_within_bounds(self, min_v, max_v):
        Z = self.params.get('inducing_inputs').data().asnumpy()
        Z[:,0] = np.clip(Z[:,0], min_v[0], max_v[0])
        Z[:,1] = np.clip(Z[:,1], min_v[1], max_v[1])
        self.params.get('inducing_inputs').set_data(Z)

    def log_likelihood_u(self, F, mu, L, Ls):
        D = mu.shape[0]
        M = L.shape[1]
        # KL_u
        Lbrod = F.expand_dims(L, axis=0)
        Lbrod = F.broadcast_axis(Lbrod, axis=(Lbrod.ndim - 3), size=(Ls.shape[-3]))
        LinvLs = F.linalg.trsm(Lbrod, Ls, alpha=1.0)
        Linvmu = F.linalg.trsm(Lbrod, mx.nd.expand_dims(mu, axis=-1), alpha=1.0)
        KL_u = 0.5 * (
                F.sum(F.square(LinvLs)) +
                F.sum(F.square(Linvmu)) -
                D * M +
                D * 2 * F.linalg.sumlogdiag(L) -
                2 * F.linalg.sumlogdiag(Ls).sum()
                )
        return KL_u
    
    
    def hybrid_forward(self, F, z_p, y_p, noiseless, direct, **kwargs):
        L, Kuf, Kff, y_f, Ls = self.get_posterior_parameters(F, z_p, **kwargs)
        #mu = kwargs['mu']
        #m, v = compute_conditional_f(F, L, Kuf, Kff, mu, Ls=Ls)
        logLik_y = self.log_likelihood_y(F, y_p, L, Kuf, Kff, y_f, Ls, **kwargs) # m, v, y_p, **kwargs)
        KL_ratio = float(float(z_p.shape[0])/self.N)
        KL_u = self.log_likelihood_u(F, y_f, L, Ls)
        return logLik_y - KL_ratio * KL_u
    
    def data_changed(self, X, Y):
        self.N = len(Y)
