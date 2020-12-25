import numpy as np
import mxnet as mx
import itertools

from utils.mxnet_util import _positive_transform
import model.costs

# mxnet implementation of sparse variation gaussian process
class VAESVGP(object):
    class Gluon_logL_SVI(mx.gluon.HybridBlock):
        def __init__(self, likelihood, encoder1, encoder1_transfer, encoder2, encoder2_transfer, decoder, X, Y,
                     dtype=np.float64, prefix=None, params=None):
            super(VAESVGP.Gluon_logL_SVI, self).__init__(prefix=prefix, params=params)
            with self.name_scope():
                self.encoder1 = encoder1
                self.encoder1_transfer = encoder1_transfer
                self.encoder2 = encoder2
                self.encoder2_transfer = encoder2_transfer
                
                self.decoder = decoder
                self.likelihood = likelihood

                self.dtype = dtype
                self.soft_zero = 1e-10
                self.data_changed(X, Y)
                
        def data_changed(self, X, Y):
            self.X, self.Y = X, Y
        
        
        def hybrid_forward(self, F, X, non_labelled, Y_batch, x_cost_reconst, kl_cost, y_cost_encoder, y_cost_reconst, num_samples_latent=1, annealing=1.0, weights=1.0, w_labelled=0.5, alpha=1.0, **kwargs):
            log_l = 0.0
            z1_mean, z1_sigma = self.encoder1(X) 
            #import pdb; pdb.set_trace()
            log_pxz_l, log_pxz_ul = 0.0, 0.0
            log_kl_l, log_kl_ul_x, log_kl_ul_y = 0.0, 0.0, 0.0
            log_py_l, log_py_l_nn, log_py_ul = 0.0, 0.0, 0.0
            
            loss_l, loss_ul = 0.0, 0.0
            
            # unlabelled samples:
            if non_labelled is not None:
                N_ul = len(non_labelled)
                X_ul = X[non_labelled,:]

                z1_mean_ul, z1_sigma_ul = z1_mean[non_labelled,:], z1_sigma[non_labelled,:]

                for i in range(num_samples_latent):
                    z1_ul = self.encoder1_transfer(F, z1_mean_ul, z1_sigma_ul)
                    
                    z2_mean_ul, z2_sigma_ul = self.encoder2(z1_ul)
                    
                    z2_ul = self.encoder2_transfer(F, z2_mean_ul, z2_sigma_ul)
                    
                    X_tilde_ul  = self.decoder(z2_ul)
                    
                    if x_cost_reconst is not None:
                        log_pxz_ul = log_pxz_ul + x_cost_reconst(F, X_ul, X_tilde_ul, self.dtype, annealing=annealing)
                    if kl_cost is not None:
                        log_kl_ul_x = log_kl_ul_x + kl_cost(F, z1_mean_ul, z1_sigma_ul, z2_mean_ul, z2_sigma_ul)
                loss_ul = F.sum((log_pxz_ul - annealing*log_kl_ul_x))/num_samples_latent/N_ul 
            
            # labelled samples
            if len(Y_batch) > 0:
                N_l = len(Y_batch)
                labelled_y = [Y_batch[i][0] for i in range(len(Y_batch))]
                X_l = X[labelled_y,:]
                Y_l = [Y_batch[i][1] for i in range(len(Y_batch))]
                y_l = mx.nd.array(Y_l, dtype=self.dtype).reshape((1,-1))
                
                z1_mean_l, z1_sigma_l = z1_mean[labelled_y,:], z1_sigma[labelled_y,:]
                
                for i in range(num_samples_latent):
                
                    z1_l = self.encoder1_transfer(F, z1_mean_l, z1_sigma_l, y_l)
                    
                    z2_mean_l, z2_sigma_l = self.encoder2(z1_l)
                    
                    z2_l = self.encoder2_transfer(F, z2_mean_l, z2_sigma_l)
                    
                    X_tilde_l  = self.decoder(z2_l)
                    
                    if x_cost_reconst is not None:
                        log_pxz_l = log_pxz_l + x_cost_reconst(F, X_l, X_tilde_l, self.dtype, annealing=annealing)
                    if y_cost_reconst is not None:
                        y_cost_tmp = y_cost_reconst(F, self.likelihood, z2_l, y_l, z1_mean_l, direct=True)
                        log_py_l = log_py_l + y_cost_tmp 
                    if kl_cost is not None:
                        log_kl_l = log_kl_l + kl_cost(F, z1_mean_l, z1_sigma_l, z2_mean_l, z2_sigma_l, y_l)
                
                if y_cost_encoder is not None:
                    log_py_l_nn = log_py_l_nn + y_cost_encoder(F, z1_mean_l, z1_sigma_l, y_l )
                loss_l = F.sum(((log_pxz_l  - annealing*log_kl_l)/num_samples_latent +alpha*log_py_l_nn)/N_l*weights[labelled_y]) + log_py_l/num_samples_latent/N_l             
                
                
            return -loss_l*w_labelled -loss_ul*(1.0-w_labelled)
            
            
        def predict_mx(self, x, diag=False, push=True):
        
            D = x.shape[-1]
            if D >= self.X.shape[-1]:
                z1_mean, z1_sigma = self.encoder1(x)
                z2_mean_, z2_sigma = self.encoder2(z1_mean)
                z2_mean = self.encoder2_transfer(mx.nd, z2_mean_, z2_sigma)
            else:
                z2_mean, z1_sigma = x, mx.nd.zeros(x.shape, dtype=self.dtype)
            if push:
                x_tilde = self.decoder.push(mx.nd, self.decoder(z2_mean))
            else:
                x_tilde = self.decoder(z2_mean)
            if type(x_tilde) is tuple:
                x_tilde = x_tilde[0]
            m, v = self.likelihood.predict(z2_mean, diag=diag)
            
            return m.reshape(-1), v, x_tilde, z2_mean, z1_sigma

        def predict(self, x, diag=False):
            m, v, x_tilde, z_mean, z_sigma = self.predict_mx(x, diag=diag)
            return m.asnumpy(), v.asnumpy(), x_tilde.asnumpy(), z_mean.asnumpy(), z_sigma.asnumpy()   

    @staticmethod
    def give_corners(bounds):
        if len(bounds) > 1:
            corners = VAESVGP.give_corners(bounds[1:])
            firsts = np.c_[np.ones((corners.shape[0],1))*bounds[0][0], corners]
            seconds = np.c_[np.ones((corners.shape[0],1))*bounds[0][1], corners]
            return np.r_[firsts, seconds]
        else:
            return np.array(bounds[-1]).reshape((-1,1))

    def set_inducing_to_data(self):
        if len(self.Y)>0:
            X = self.X
            inds = [self.Y[i][0] for i in range(len(self.Y)) ]
            Y_passed = np.array([self.Y[i][1] for i in range(len(self.Y)) ])
            X = X[inds,:]        
            X = self.predict(mx.nd.array(X, dtype=self.dtype), diag=True)[3]
            
            self.likelihood.set_inducing_to_data(X, Y=Y_passed)
        else:
            return
    
    def __init__(self, X, Y, likelihood, encoder1, encoder1_transfer, encoder2, encoder2_transfer, decoder, d_latent,
                 num_samples_latent=50, num_inducing=10, batch_size=20, ctx=mx.cpu(),
                 dtype=np.float64, name='gp', debug=False, y_lik_approx="lb",  act='relu'):
        self.name = name
        self.ctx = likelihood.ctx
        self.dtype = dtype
        self.num_samples_latent = num_samples_latent
        self.likelihood = likelihood
        self.batch_size = batch_size
        self.d_latent = d_latent #encoder1.dim_latent
        
        self.encoder1 = encoder1 
        self.encoder1_tranfer = encoder1_transfer
        self.encoder2 = encoder2
        self.encoder2_tranfer = encoder2_transfer
        self.decoder = decoder
        
        self.encoder1.initialize(ctx=ctx)
        self.encoder2.initialize(ctx=ctx)
        self.decoder.initialize(ctx=ctx)
        self.X_try = None
        self.set_XY(X,Y)

    @property
    def Y_direct(self):
        return np.array([self.Y[i][1] for i in range(len(self.Y))])

    def set_XY(self, X, Y):
        self.X = mx.nd.array(X, ctx=self.ctx, dtype=self.dtype)
        self.Y = Y
        #self.labelled_ind = self.likelihood.get_labelled_ind(self.Y)
        self.likelihood.data_changed(X, Y)
        self.set_functions()
        self.set_inducing_to_data()

    def set_functions(self):
        self._comp_logL = self.Gluon_logL_SVI(self.likelihood, self.encoder1, self.encoder1_tranfer, self.encoder2, self.encoder2_tranfer, self.decoder,
                                              self.X, self.Y, dtype=self.dtype,
                                              prefix=self.name+'_logL_', params=self.likelihood.params)
    
    
    def get_labels(self, X, Y, all_data=True):
        tot_len = X.shape[0]
        
        inds_used_y = [i for i in range(len(Y))]

        if all_data:
            # Find indices with observations:
            yind = [Y[i][0] for i in range(len(Y))]            
            inds_not_labelled = np.ones((tot_len,), dtype=bool)
            inds_not_labelled[yind] = False
            inds_not_used = np.where(inds_not_labelled)[0].reshape(-1)
        else:
            inds_not_used = []
        return inds_not_used, inds_used_y
        
    
    def get_batch_from_inds(self, X, Y, inds_not_used, inds_y):
        Y_batch = [Y[i] for i in inds_y.asnumpy()] if inds_y is not None else []
        return inds_not_used, Y_batch

    def do_optimization(self, x_cost_reconst=None, kl_cost=None, y_cost_encoder=None, y_cost_reconst=None, max_iters=100, all_data=False, print_iter=False, step_rate=1e-3, annealing=True, weighting = False):
        
        X, Y = self.X, self.Y
        if weighting:
            all_data = False
    
        schedule = mx.lr_scheduler.FactorScheduler(step=300, factor=0.5)
        trainer = mx.gluon.Trainer(self.collect_params(likelihood=True), 'adam', {'learning_rate': step_rate, 'lr_scheduler':schedule})  
        
        for p in self.likelihood.kern.params.items():
            p[1].lr_mult  = 0.1/step_rate
        self.likelihood.collect_params()['svgp_sigma2'].lr_mult = 0.1/step_rate
        self.likelihood.collect_params()['svgp_inducing_inputs'].lr_mult = 1e-5/step_rate
        
        losses = []
        
        
        inds_not_used, inds_used_y = self.get_labels(X, Y, all_data=all_data)
        inds_all = [inds_not_used, inds_used_y]
        #import pdb; pdb.set_trace()
        
        lens = [len(inds) for inds in inds_all if len(inds)>0]
        batch_size_max =  np.min([self.batch_size, np.max(lens)])
        lens_all = np.sum(lens)
        num_iter = np.min([min(lens), int(lens_all/batch_size_max)])
        batch_sizes = [ int(np.ceil(len(inds)/num_iter)) for inds in inds_all]
        e = 0
        ei = 0
        annealing_factor =1.0
        
        W = mx.nd.ones(X.shape[0], dtype=self.dtype)
        
        while e < max_iters:
                
            if annealing and max_iters > 499:
                annealing_factor = max(e / 200.0 - 0.5, 0)
                annealing_factor = min(annealing_factor, 1.0)

            X_, Y_ = X, Y

            data_loaders = [(mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset(mx.nd.array(inds, dtype=int)), batch_size=batch_sizes[ind_i], shuffle=True) if len(inds)>0 else [None]*num_iter) for ind_i,inds in enumerate(inds_all) ]
            L_e = 0
            
            i = 0
            failed = False
            for i, (data_batch_not_used, data_batch_used_y)  in enumerate(zip(*data_loaders)):
                inds_not_used, Y_batch = self.get_batch_from_inds(X_, Y_, data_batch_not_used, data_batch_used_y)
                with mx.autograd.record():
                    loss = self._comp_logL(X_, inds_not_used, Y_batch,  x_cost_reconst, kl_cost, y_cost_encoder, y_cost_reconst, self.num_samples_latent, annealing_factor, W)
                    loss.backward()
                if not np.isnan(loss.asscalar()) and not np.isinf(loss.asscalar()):
                    trainer.step(batch_size=batch_size_max, ignore_stale_grad=True)
                else:
                    failed = True
                    break
                self.likelihood.update_sensitive_parameters()

                L_e += -loss.asscalar()/batch_size_max
            losses += [L_e]
            if not failed:
                e += 1
            ei += 1
            if ei % 10 == 0:
                print("Iter {} (try {}), loss {}".format(e, ei, L_e))
        print('-'*50)
        print("epoch {} Average logL: {}.".format(e+1, L_e/(i+1)))
        print("GP parameters updated to:")
        print("lengthscale:")
        print(_positive_transform(mx.nd, self.likelihood.kern.params.get('lengthscale').data()).asnumpy())
        print("variance:")
        print(_positive_transform(mx.nd, self.likelihood.kern.params.get('variance').data()).asnumpy())
        print("noise:")
        print(self.likelihood.sigma2s[0])
        print('-'*50)
            
        return losses


    def optimize(self, max_iters=100, step_rate=1e-2, x_cost_reconst=None, kl_cost=None,\
                 y_cost_encoder=None, y_cost_reconst=None, all_data=True, print_iter=False,\
                 freeze_vae=False, freeze_likelihood=False, annealing=False, weighting=False):
        
        if freeze_vae:
            print("Freezing VAE")
            self.set_grad_update_mode(self.encoder1.collect_params().values(), 'null')
            self.set_grad_update_mode(self.encoder2.collect_params().values(), 'null')
            self.set_grad_update_mode(self.decoder.collect_params().values(), 'null')

        print("Optimizing with cost functions:\n{}\n{}\n{}\n{}\n{}".format(x_cost_reconst, kl_cost, y_cost_encoder, y_cost_reconst, all_data))
        self.do_optimization(x_cost_reconst=x_cost_reconst, kl_cost=kl_cost, y_cost_encoder=y_cost_encoder,y_cost_reconst=y_cost_reconst, max_iters=max_iters, all_data=all_data, print_iter=print_iter, step_rate=step_rate, annealing=annealing, weighting=weighting)

        self.set_grad_update_mode(self.encoder1.collect_params().values(), 'write')
        self.set_grad_update_mode(self.encoder2.collect_params().values(), 'write')
        self.set_grad_update_mode(self.decoder.collect_params().values(), 'write')


    def set_grad_update_mode(self, params, mode):
        for param in params:
            param.grad_req = mode
    @property
    def X_latent(self):
        return self.predict(self.X)[3]
        
    def predict_noiseless(self,x, full_cov=False):
        x = mx.nd.array(x, dtype=self.dtype)
        m, v = self.likelihood.predict(x, diag=(not full_cov))
        return m.asnumpy(), v.asnumpy()

    def collect_params(self, likelihood=False):
        ret = mx.gluon.ParameterDict(self.name + '_')
        if likelihood:
            ret.update(self.likelihood.collect_params())
        ret.update(self.encoder1.collect_params())
        ret.update(self.encoder2.collect_params())
        ret.update(self.decoder.collect_params())
        return ret

    def predict(self, Xnew, diag=False):
        return self._comp_logL.predict(Xnew, diag=diag)