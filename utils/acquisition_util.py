import numpy as np
from typing import Callable, List, Tuple, Dict

class AcquisitionFunction():
    """
    This class implements the general AcquisitionFunction
    
    :param options: Dictionary containing the acquisition function options
    :param optimizer_options: Dictionary containing options for the acquisition function optimizer
    """
    def __init__(self, options: Dict={}, optimizer_options: Dict={}):
        self.optimizer_options = optimizer_options
        self.pool_size = options.get('pool_size', -1)
        self.acq_samples = options.get('acq_samples', 500)
        self.acq_opt_restarts = options.get('acq_opt_restarts', 10)

    def acq_fun_optimizer(self, m, bounds: np.ndarray, batch_size: int, get_logger: Callable) -> np.ndarray:
        """
        Implements the optimization scheme for the acquisition function
        
        :param m: The model which posterior is used by the acquisition function (from which the samples are drawn from)
        :param bounds: the optimization bounds of the new sample
        :param batch_size: How many points are there in the batch
        :param get_logger: Function for receiving the legger where the prints are forwarded.
        :return: optimized locations
        """
        raise NotImplementedError
    
    def reset(self, model) -> None:
        """
        Some acquisition functions need to be reseted, this method is for that.
        :param model: the model to be passed to the acquisition function (some acquisition functions need a model at this point)
        """
        None


class RandomAcquisitionFunction(AcquisitionFunction):
    """
    This class implements the general AcquisitionFunction
    
    :param options: Dictionary containing the acquisition function options
    :param optimizer_options: Dictionary containing options for the acquisition function optimizer
    """
    def evaluate(self, x: np.ndarray, model) -> np.ndarray:
        """
        Computes the Expected Improvement.
        :param x: points where the acquisition is evaluated.
        :param m: the GP model which posterior is used
        :return: acquisition function value
        """
        return np.random.rand(x.shape[0],1)

    def evaluate_with_gradients(self, x: np.ndarray, model) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Expected Improvement.
        :param x: points where the acquisition is evaluated.
        :param model: the GP model which posterior is used
        :return: A tuple containing the acquisition function values and their gradients
        """
        return np.random.rand(x.shape[0],1), np.zeros_like(x)
