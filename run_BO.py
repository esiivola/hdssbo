import numpy as np
import scipy
import mxnet as mx
import emukit
import sys
import os
import copy 
import functools
from functools import partial
from contextlib import redirect_stdout

from utils.mxnet_util import _positive_transform, make_diagonal, extract_diagonal, _positive_transform_reverse, make_stdcdf
from utils.optimization_constraints import WithinEllipsoid, WithinConvexSet, WithinStableSet
from utils.acquisition_util import RandomAcquisitionFunction
from utils import plotting_util
from utils import run_util
from utils import parameter_util


from model.kernel import RBF, Add, Matern32, Matern52
from model.vaesvgp import VAESVGP
from model import observation_models
from model import emukit_wrappers

from datasets import datasets


from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core.initial_designs import RandomDesign
from emukit.core.constraints import InequalityConstraint, LinearInequalityConstraint
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement, ProbabilityOfImprovement, NegativeLowerConfidenceBound
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.core.optimization.anchor_points_generator import ObjectiveAnchorPointsGenerator
from emukit.examples.preferential_batch_bayesian_optimization.pbbo.acquisitions import ThompsonSampling, EmukitAcquisitionFunctionWrapper 


# Some emukit method need to be modified a bit:
#############################################################->
class NonlinearInequalityConstraint(InequalityConstraint):
    """
    Constraint of the form lower_bound < g(x) < upper_bound
    """
    def __init__(self, constraint_function, lower_bound: np.ndarray, upper_bound: np.ndarray,
                 jacobian_fun=None):
        """
        :param constraint_function: function defining constraint in b_lower < fun(x) < b_upper.
                                    Has signature f(x) -> array, shape(m,) where x is 1d and m is the number of constraints
        :param lower_bound: Lower bound vector of size (n_constraint,). Can be -np.inf for one sided constraint
        :param upper_bound: Upper bound vector of size (n_constraint,). Can be np.inf for one sided constraint
        :param jacobian_fun: Function returning the jacobian of the constraint function. Optional, if not supplied
                             the optimizer will use finite differences to calculate the gradients of the constraint
        """

        super().__init__(lower_bound, upper_bound)

        self.fun = constraint_function
        self.jacobian_fun = jacobian_fun

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate whether constraints are violated or satisfied at a set of x locations
        :param x: Array of shape (n_points x n_dims) containing input locations to evaluate constraint at
        :return: Numpy array of shape (n_input,) where an element will be 1 if the corresponding input satisfies the
                 constraint and zero if the constraint is violated
        """
        fun_x = np.array([self.fun(x) for x in x])
        return np.all([np.all(fun_x >= self.lower_bound, axis=1), np.all(fun_x <= self.upper_bound,axis=1)], axis=0)

emukit.core.constraints.NonlinearInequalityConstraint = NonlinearInequalityConstraint

def _get_scipy_constraints(constraint_list):
    """
    Converts list of emukit constraint objects to list of scipy constraint objects
    :param constraint_list: List of Emukit constraint objects
    :return: List of scipy constraint objects
    """

    scipy_constraints = []
    for constraint in constraint_list:
        if isinstance(constraint, NonlinearInequalityConstraint):
            if constraint.jacobian_fun is None:
                # No jacobian supplied -> tell scipy to use finite difference method
                jacobian = '2-point'
            else:
                # Jacobian is supplied -> tell scipy to use it
                jacobian = constraint.jacobian_fun

            scipy_constraints.append(
                scipy.optimize.NonlinearConstraint(constraint.fun, constraint.lower_bound, constraint.upper_bound,
                                                   jacobian))
        elif isinstance(constraint, LinearInequalityConstraint):
            scipy_constraints.append(scipy.optimize.LinearConstraint(constraint.constraint_matrix,
                                                                     constraint.lower_bound,
                                                                     constraint.upper_bound))
        else:
            raise ValueError('Constraint type {} not recognised'.format(type(constraint)))
    return scipy_constraints
emukit.core.optimization.optimizer._get_scipy_constraints = _get_scipy_constraints

def evaluate(self, x: np.ndarray) -> np.ndarray:
    """
    Evaluate whether constraints are violated or satisfied at a set of x locations
    :param x: Array of shape (n_points x n_dims) containing input locations to evaluate constraint at
    :return: Numpy array of shape (n_points, ) where an element will be 1 if the corresponding input satisfies all
                constraints and zero if any constraint is violated
    """
    if self.constraint_matrix.shape[1] != x.shape[1]:
        raise ValueError('Dimension mismatch between constraint matrix (second dim {})' +
                        ' and input x (second dim {})'.format(self.constraint_matrix.shape[1], x.shape[1]))

    # Transpose here is needed to handle input dimensions
    # that is, A is (n_const, n_dims) and x is (n_points, n_dims)
    ax = self.constraint_matrix.dot(x.T).T
    return np.all((ax >= self.lower_bound) & (ax <= self.upper_bound), axis=1)

LinearInequalityConstraint.evaluate = evaluate
#############################################################<-



def run(ind):
    #Runs the BO routine 
    
    # Get the config file
    save_name, config = run_util.get_setting_from_index(ind)
    np.random.seed(config.get('SEED', parameter_util.SEED))

    # Get the VAE + GP models to be used
    print("Get {}-dimensional VAE for BO".format(config.get('DIM_LATENT', parameter_util.DIM_LATENT)))
    model_opt, opt_init, opt_loop, config = run_util.get_lvae_and_opts(config)
    
    # Get run parameters based on what is in config, if the parameter value not found, the default value is used.
    np.random.seed(config.get('SEED', parameter_util.SEED)) # random seed
    plot = config.get('PLOT', parameter_util.PLOT) # if we plot the latent space
    init_size = config.get('INIT_SIZE', parameter_util.INIT_SIZE) # number of initial observations
    dim_latent = config.get('DIM_LATENT', parameter_util.DIM_LATENT) # the dimensionality of the latent space
    dataset_id = config.get('DATASET', parameter_util.DATASET) # the dataset & black box function to be used
    n_iter = config.get('N_ITER', parameter_util.N_ITER) # number of BO iterations
    acq = config.get('ACQ', parameter_util.ACQ) # acquisition function that is used
    optimize_rate = config.get('OPTIMIZE_RATE', parameter_util.OPTIMIZE_RATE) # a parameter defining how often the latent space is updated (ar VAE parameters are updated)
    optimized_space =  config.get('OPTIMIZED_SPACE', parameter_util.OPTIMIZED_SPACE) # Acquisition function optimization restriciton method
    
    print("Run will be saved with name:")
    print(save_name)
    

    # Get the dataset (unlabelled values)
    data = run_util.get_dataset(dataset_id, config)
    
    # Get the blaack box function
    cost_function =  data.get_objective()
    
    # Sample initial observations
    print("Sampling {} initial samples".format(init_size))
    X_init =  model_opt.X
    np.random.seed(config.get('SEED', parameter_util.SEED))
    
    
    x_init = run_util.sample(init_size, data=data.get_data_pure())
    y_init = [ (X_init.shape[0] + i, yi[0]) for i, yi in enumerate(cost_function(x_init))]

    X_full = np.r_[model_opt.X.asnumpy(), x_init]
    Y_full = model_opt.Y + y_init
    
    #Initialize the model with the initial values 
    model_opt.set_XY(X_full, Y_full)
    # Initialize the locations of the inducing points
    model_opt.set_inducing_to_data()

    print("Optimizing model with INIT setting")
    #Optimize the GP and VAE parameters based on configuration
    model_opt.optimize(**opt_init[-1].copy() )
    
    if plot: # Plot the latent space if configured to do so
        y_tr_latent =  cost_function(model_opt.X) # black box function values for the all of training data
        H_mean_tr = model_opt.predict(X_init)[3] # locations of the training data in the latent space
        samples_l = model_opt.predict(mx.nd.array(x_init, dtype=model_opt.dtype) )[3] # locations of the labelled samples in the latent space
        
        plotting_util.plot_latent_space_direct(model_opt, H_mean_tr, y_tr_latent.flatten(), cost_function=cost_function_, savename=save_name[:-4] + "-init.png", samples_l = samples_l, config=config)
        
    for k in range(len(opt_init)):
        model_opt.set_inducing_to_data()
        model_opt.optimize(**opt_init[k].copy() )
    
    if plot: # plot the latent space after having trained the latent space
        H_mean_tr = model_opt.predict(X_init)[3]
        samples_l = model_opt.predict(mx.nd.array(x_init, dtype=model_opt.dtype) )[3]
        plotting_util.plot_latent_space_direct(model_opt, H_mean_tr, y_tr_latent.flatten(), cost_function=cost_function_, savename=save_name[:-4] + "-loop_0.png", samples_l = samples_l, inducing=True, config=config)  
 
    wrapped_model_opt = emukit_wrappers.VAEGPLVMModelWrapper(model_opt) # wrpa the model inside a emukit wrapper so that it can be used in BO
    
    with open(save_name, 'w+') as f: # Save the initial observations
        #y_init = y_init.flatten()
        y_init_clean = np.array([y_init[j][1] for j in range(len(y_init))]).reshape(-1)
        for i in range(y_init_clean.shape[0]) :
            f.write(str(y_init_clean[i]) + '\n')
    
    print("Starting BO loop which is ran for {} iterations".format(n_iter))
    
    space = None
    
    for i in range(n_iter): # run the BO for n_iter iterations
        print("Iteration {}".format(i))
        if space is None: # If the latent space has changed, the acquisition function optimization restriction method has to be updated
            print("Recomputing parameter space edges")
            train_data_small = wrapped_model_opt.model.predict(model_opt.X)[3]
            lims = np.array([[min(train_data_small[:,i]), max(train_data_small[:,i])] for i in range(dim_latent) ]) # The hyper cube restricting the data

            space_array = []
            for j in range(dim_latent):
                space_array += [ContinuousParameter('x{}'.format(j), lims[j,0], lims[j,1])]
    
            # Decide the optimization bounds
            if optimized_space == parameter_util.CONVEX:
                lims_nonlinear = WithinConvexSet(train_data_small)
                constraint = NonlinearInequalityConstraint(lims_nonlinear.f, lims_nonlinear.bounds[0], lims_nonlinear.bounds[1], jacobian_fun= lambda x: lims_nonlinear.constraint_matrix) #LinearInequalityConstraint(lims_nonlinear.jacobian(None), lims_nonlinear.bounds[0], lims_nonlinear.bounds[1])
                space = ParameterSpace(space_array, [constraint])
            elif optimized_space == parameter_util.ELLIPS:
                lims_nonlinear = WithinEllipsoid(train_data_small)
                constraint = NonlinearInequalityConstraint(lims_nonlinear.f, lims_nonlinear.bounds[0], lims_nonlinear.bounds[1], jacobian_fun= lambda x: lims_nonlinear.constraint_matrix) #, hessian_fun=lims_nonlinear.hessian)
                space = ParameterSpace(space_array, [constraint])
            elif optimized_space == parameter_util.STABLE:
                lims_nonlinear = WithinStableSet(train_data_small,  wrapped_model_opt.model)
                constraint = NonlinearInequalityConstraint(lims_nonlinear.f, lims_nonlinear.bounds[0], lims_nonlinear.bounds[1], jacobian_fun= lambda x: lims_nonlinear.constraint_matrix)
                space = ParameterSpace(space_array, [constraint])
            else:
                space = ParameterSpace(space_array)
            
            
            # Decide the acquisition function
            if acq == parameter_util.TS:
                acquisition = ThompsonSampling()
            elif acq == parameter_util.QEI:
                emukit_acquisition = ExpectedImprovement(wrapped_model_opt)    
            elif acq == parameter_util.LCB:
                emukit_acquisition = NegativeLowerConfidenceBound(wrapped_model_opt)
            elif acq == parameter_util.PI:
                emukit_acquisition = ProbabilityOfImprovement(wrapped_model_opt)
            else:
                acquisition = RandomAcquisitionFunction()
            
            
            emukit_acquisition = EmukitAcquisitionFunctionWrapper(wrapped_model_opt, acquisition)
            acquisition_optimizer = GradientAcquisitionOptimizer(space)
            
        # Create the BO object and run it for one iteration
        bo = BayesianOptimizationLoop(model=wrapped_model_opt, space=space, acquisition=emukit_acquisition, acquisition_optimizer=acquisition_optimizer) 
        print("Running BO loop once")
        cost_function_latent = partial(cost_function, model_opt=model_opt) 
        
        bo.run_loop( cost_function_latent, 1)
        
        print("New observation: {}".format( model_opt.Y[-1:]))
        
        # Optimize the model as the new observation has been received
        model_opt.optimize(**opt_loop[0])
        if (i+1) % optimize_rate == 0: # optimize the latent space every optimize_rate iteration
            print("relearning the latent space")
            if len(opt_loop)>1:
                space = None
            for ii in range(1,len(opt_loop)):
                model_opt.set_inducing_to_data()
                model_opt.optimize(**opt_loop[ii])

        
        # Save the results
        with open(save_name, 'a') as f:
            y_s_ = [yi[1] for yi in model_opt.Y[-1:]]
            for j in range(len(y_s_)):
                f.write(str(y_s_[j]) + '\n')
        
        if plot:
            H_mean_tr = model_opt.predict(X_init)[3]
            samples_l = model_opt.predict(model_opt.X)[3]
            samples_l = samples_l[-i-1-x_init.shape[0]:,:]
            plot_latent_space_direct(model_opt, H_mean_tr, y_tr_latent.flatten(), cost_function=cost_function_, savename=save_name[:-4] + "-loop_{}.png".format(i+1), samples_l = samples_l, highlight=samples_l[-1:,:].reshape((1,-1)), config = config)

    print("All done, exiting")
    for j in range(10):
        print("\n")
    print("---------------------------------------------------------")

def run_robust_with_redirect(ind, logfile):
    with open(logfile, 'w') as f:
        with redirect_stdout(f):
            try:
                run(ind)
            except Exception as e:
                print(e)

def run_robust(ind):
    try:
        run(ind)
    except Exception as e:
        print(e)
            
            
if __name__ == "__main__":
    if len(sys.argv) > 1:
        ind = int(sys.argv[1])
        run(ind)
    else:
        logpath = 'vpbbo_logs/'
        names, scripts = run_util.get_configs() # Get all different configurations defined in parameter util
        print("{} jobs in total".format(len(names)))
        for i in range(len(names)):
            logs = logpath+str(i)+".out"
            print("Running {}".format(i))
            run_robust(i)
