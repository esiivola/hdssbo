import numpy as np
import mxnet
from utils.mxnet_util import sample_Gaussian

def encoder1_transfer_cartesian(F, z_mean, z_sigma, y=None):
    return sample_Gaussian(F, z_mean, z_sigma)

def encoder2_transfer_cartesian(F, z_mean, z_sigma, y=None):
    return z_mean

def kl_cost_cartesian(F, z1_mean, z1_sigma, z2_mean, z2_sigma, y=None):
    return kl_cost_normal(F, z1_mean, z1_sigma, p_mu=0, p_sigma=1)

def y_cost_encoder(F, z_mean, z_sigma, y):
    return 0.0

def y_cost_reconst(F, likelihood, z, y, z_mean, direct=True, **kwargs):
    return likelihood(z, y, z_mean, direct, **kwargs)

def log_pdf_normal(F, y, mean, sigma):
    N = y.shape[0]
    return F.reshape( -F.log(sigma) - 0.5*F.square((y-mean)/sigma) - np.log(2*np.pi)/2, shape=-1 ) # F.sum( -F.log(sigma) - 0.5*F.square((y-mean)/sigma) ) #

def y_cost_gp_pure(F, likelihood, x, y):
    return likelihood(x, y, None, None)

def kl_cost_normal(F, q_mu, q_sigma, p_mu=0, p_sigma=1):
    if p_mu is None:
        p_mu = q_mu
    else:
        p_mu = p_mu*F.ones_like(q_mu)
    p_sigma = p_sigma*F.ones_like(q_sigma)
    KL = 0.5*(  2.0*F.log(p_sigma/q_sigma) + (F.square(q_sigma) + F.square(q_mu-p_mu) )/F.square(p_sigma) - 1 )
    return F.reshape(F.sum(KL, axis=1), shape=-1) #  #
