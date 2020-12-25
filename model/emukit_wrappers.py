import numpy as np
import emukit
import emukit.core.interfaces
import mxnet as mx

from typing import Tuple

from model.vaesvgp import VAESVGP

class VAEGPLVMModelWrapper(emukit.core.interfaces.IModel, emukit.core.interfaces.IDifferentiable):
    """
    This is a wrapper around vaegplvm model
    """
    def __init__(self, vaegplvm_model, opt_options=None, max_value=0, from_sampled=False):
        """
        :param  vaegplvm_model: model object to wrap
        """
        self.model = vaegplvm_model
        self.opt_options = opt_options
        self.distribution_lb = None
        self.kde = None
        self.max_value = max_value
        self.from_sampled = from_sampled

    def predict(self, X: np.ndarray, full_cov=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param X: (n_points x n_dimensions) array containing locations at which to get predictions
        :return: (mean, variance) Arrays of size n_points x 1 of the predictive distribution at each input location
        """
        if len(X.shape)>2:
            X = X[0,:,:]
        mean, variance = self.model.predict(mx.nd.array(X, dtype = self.model.dtype), diag=not full_cov)[:2]
        mean = mean.reshape(-1,1)
        variance = variance.reshape(mean.shape[0],-1)
            
        return mean, variance
    
    def get_prediction_gradients(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_mx = mx.nd.array(X, dtype = self.model.dtype)
        X_mx.attach_grad()
        
        with mx.autograd.record():
            mean, variance = self.model._comp_logL.predict_mx(X_mx, diag=True, push=False)[:2]
            mean_g = mx.autograd.grad(mean, X_mx, retain_graph=True)
            var_g = mx.autograd.grad(variance, X_mx)
        mg, vg = mean_g[0].asnumpy().reshape(X.shape[0],-1), var_g[0].asnumpy().reshape(X.shape[0],-1)
        return mg, vg
    
    def _project_to_sampled(self, X):
        if self.from_sampled:
            X_projected = self.model.predict(self.model.X)[3]
            dist = np.sqrt(np.sum((X_projected[:,None,:] - X[None,:,:])**2, axis=-1))
            inds = [np.argmin(dist[:,i].flatten()) for i in range(dist.shape[1])]
            X_new = self.model.X.asnumpy()[inds,:] 
        else:
            X_new = self.model.predict(mx.nd.array(X, dtype=self.model.dtype))[2]
        return X_new

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Sets training data in model
        :param X: New training features
        :param Y: New training outputs

        It is assumed that if the data to be set is larger
        than the current size of the model, only the last
        items in the data to be added exceeding the size of
        the current model are added.
        
        Also, if the point is given in the latent space, closest match in the unlabelled data is added instead
        """
        if X.shape[0]>0:
            X = X[0,:].reshape((1,-1))
            Y = Y[0,:].reshape((-1))
            ind = self.model.X.shape[0]
            X_set = self.model.X.asnumpy()
            y_set = self.model.Y
            self.model.X_try =  X
            X = self._project_to_sampled(X)
            X_set = np.concatenate((X_set, X), axis=0)
            y_set += [(i+ind, Y[i]) for i in range(len(Y))]
            self.model.set_XY(X_set,y_set)

    def optimize(self, prnt=False, x_cost=True, y_cost=True, freeze_encoder=False, all_data=True, opt_iter=5, weighted_cost=True):
        """
        Optimizes model hyper-parameters
        """
        if self.opt_options is not None:
            for opt_option in self.opt_options:
                self.model.optimize(**opt_option)

    @property
    def X(self) -> np.ndarray:
        """
        :return: An array of shape n_points x n_dimensions containing training inputs
        """
        
        return np.empty((0,self.model.d_latent)) #self.model.predict(self.model.X)[3]

    @property
    def Z(self) -> np.ndarray:
        """
        :return: An array of shape n_points x n_dimensions containing training inputs
        """
        return self.model.likelihood.inducing_points

    @property
    def posterior(self) -> np.ndarray:
        return self.model.likelihood.posterior
    
    @property
    def kern(self):
        return self.model.likelihood.get_GPy_kern()
    
    @property
    def sigma2s(self) -> np.ndarray:
        return self.model.likelihood.sigma2s
    
    def get_current_best(self):
        return self.model.likelihood.get_minimum()
    
    @property
    def Y(self) -> np.ndarray:
        """
        :return: An array of shape n_points x 1 containing training outputs
        """
        return np.empty((0,1))
