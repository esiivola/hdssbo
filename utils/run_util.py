import numpy as np
import mxnet as mx

import os
import sys
import itertools

import random
import emukit
from mxnet.initializer import Xavier

from model import observation_models
from model import emukit_wrappers
from model import costs
from model import mlp 
from utils import parameter_util
from model import kernel 
from model import vaesvgp
from datasets import datasets


def get_lvae_and_opts(config):
    config = {**config, **parameter_util.METHODS[config.get('METHOD', parameter_util.METHOD)]}
    dataset_id = config.get('DATASET', parameter_util.DATASET)
    dataset = get_dataset(dataset_id, config)
    n_init = config.get('N_INIT', parameter_util.N_INIT)
    n_loop = config.get('N_LOOP', parameter_util.N_LOOP)
    step_rate = config.get('STEP_RATE', parameter_util.STEP_RATE)
    freeze_vae = config.get('FREEZE_VAE', parameter_util.FREEZE_VAE)
    all_data = config.get('ALL_DATA', parameter_util.ALL_DATA)
    
    x_cost_reconst, kl_cost, y_cost_reconst, y_cost_encoder  = get_model_costs(dataset)
    
    if config.get('METHOD', parameter_util.METHOD) == 'deepgp':
        config['FROM_SAMPLED'] = True
    
    
    if freeze_vae:
        kl_cost = None
        x_cost_reconst = None
        y_cost_encoder = None

    dim_latent = config.get('DIM_LATENT', parameter_util.DIM_LATENT)
    model = get_trained_model(dim_latent=dim_latent, dataset=dataset, config=config)
    
    opt_init = [{'max_iters':n_init, 'step_rate':step_rate, 'y_cost_encoder': None, 'y_cost_reconst':y_cost_reconst, 'x_cost_reconst':None, 'kl_cost':None, 'freeze_vae':True, 'freeze_likelihood':False, 'all_data':False, 'print_iter':True, 'annealing':True}]


    opt_loop = [{'max_iters':n_init, 'step_rate':step_rate, 'y_cost_encoder': None, 'y_cost_reconst':y_cost_reconst, 'x_cost_reconst':None, 'kl_cost':None, 'freeze_vae':True, 'freeze_likelihood':False, 'all_data':False, 'print_iter':True, 'annealing':True},
                {'max_iters':n_loop, 'step_rate':step_rate, 'y_cost_encoder': y_cost_encoder, 'y_cost_reconst':y_cost_reconst, 'x_cost_reconst':x_cost_reconst, 'kl_cost':kl_cost, 'freeze_vae':freeze_vae, 'freeze_likelihood':False, 'all_data':all_data, 'print_iter':True, 'annealing':True, 'step_rate':step_rate},
                {'max_iters':n_init, 'step_rate':step_rate, 'y_cost_encoder': None, 'y_cost_reconst':y_cost_reconst, 'x_cost_reconst':None, 'kl_cost':None, 'freeze_vae':True, 'freeze_likelihood':False, 'all_data':False, 'print_iter':True, 'annealing':True}]
    
    return model, opt_init, opt_loop, config

def get_configs(options=parameter_util.ALTERNATIVES):
    names = []
    configs = []
    ARRAYS = [alt[1] for alt in options]
    NAMES = [alt[0] for alt in options]
    for sizes in itertools.product(*ARRAYS):
        name_i = parameter_util.RES_PATH
        config_i = {}
        for size_i, size in enumerate(sizes):
            config_i[NAMES[size_i]] = size
        for key in config_i.keys():
            name_i += '{}_{}-'.format(key, config_i[key])
        names += [name_i[:-1]+'.txt']
        configs += [config_i]
    return names, configs    


def get_setting_from_index(ind):
    names, configs = get_configs()
    print("Number of different options: {}".format(len(names)))
    return names[ind], configs[ind]


def get_model_costs(data):
    x_cost_reconst = data.get_x_cost_reconst()
    y_cost_reconst = costs.y_cost_reconst
    kl_cost = costs.kl_cost_cartesian
    y_cost_encoder = costs.y_cost_encoder
    return x_cost_reconst, kl_cost, y_cost_reconst, y_cost_encoder

def get_model_likelihood(dim_latent, N, config):
    fix_sigma = config.get('FIX_SIGMA', parameter_util.FIX_SIGMA)
    num_inducing = config.get('NUM_INDUCING', parameter_util.NUM_INDUCING)
    lengthscale=config.get('LENGTHSCALE', parameter_util.LENGTHSCALE)
    variance=config.get('VARIANCE', parameter_util.VARIANCE)    
    ard=config.get('ARD', parameter_util.ARD)
    fix_kern = config.get('FIX_KERN', parameter_util.FIX_KERN) 
    dtype = config.get('DTYPE', parameter_util.DTYPE)
    kern = kernel.RBF(dim_latent, ARD=ard, lengthscale=lengthscale, variance=variance, dtype=dtype, name='rbf', fix=fix_kern)
    likelihood = observation_models.SVGP(kern, dim_latent, N, fix_sigma=fix_sigma, num_inducing=num_inducing) 

    return likelihood
    
def get_model_structure(dataset, dim_latent, dim_data, config):
    act = config.get('ACT', parameter_util.ACT)
    depth=config.get('DEPTH', parameter_util.DEPTH)
    dtype = config.get('DTYPE', parameter_util.DTYPE)
    
    data_dimensionality = dataset.data_dimensionality
    encoder1_units = [dim_data] + [dim_data for j in range(depth-2)] + [dim_latent]
    decoder_units = [dim_latent] + [dim_data for j in range(depth-2)] + [dim_data]
    
    #Build encoder:
    if data_dimensionality == parameter_util.DATA_1D:
        encoder1 = mlp.MLP_encoder1_cartesian(encoder1_units, act=act, prefix='encoder1', dtype=dtype)
        encoder1_transfer = costs.encoder1_transfer_cartesian
        encoder2 = mlp.MLP_encoder2_cartesian(prefix='encoder2')
        encoder2_transfer = costs.encoder2_transfer_cartesian
    elif data_dimensionality == parameter_util.DATA_IMAGE:
        encoder1 = mlp.MLP_encoder1_image(encoder1_units, act=act, prefix='encoder1', dtype=dtype)
        encoder1_transfer = costs.encoder1_transfer_cartesian  # costs.encoder1_transfer_cartesian # costs.encoder2_transfer_cartesian
        encoder2 = mlp.MLP_encoder2_cartesian(prefix='encoder2')
        encoder2_transfer = costs.encoder2_transfer_cartesian
    else:
        raise ValueError("Unknown data dimensionality: " + data_dimensionality)
    
    decoder = dataset.get_decoder(decoder_units, act=act, dtype=dtype)
    return encoder1, encoder1_transfer, encoder2, encoder2_transfer, decoder

def get_untrained_model(dim_latent=parameter_util.DIM_LATENT, likelihood_type=parameter_util.SVGP, dataset=None, config={}):
    
    #Get needed params from config
    num_samples_h = config.get('NUM_SAMPLES_H', parameter_util.NUM_SAMPLES_H)

    act = config.get('ACT', parameter_util.ACT)
    depth=config.get('DEPTH', parameter_util.DEPTH)
    dtype = config.get('DTYPE', parameter_util.DTYPE)
    
    #Get data
    numpy_data = dataset.get_data()
    dim_data = numpy_data.shape[-1]
    
    encoder1, encoder1_transfer, encoder2, encoder2_transfer, decoder = get_model_structure(dataset, dim_latent, dim_data, config)
    likelihood = get_model_likelihood(dim_latent, numpy_data.shape[0], config)

    Y = []
    m = vaesvgp.VAESVGP(numpy_data, Y, likelihood, encoder1, encoder1_transfer, encoder2, encoder2_transfer, decoder, dim_latent,
                 num_samples_latent=num_samples_h, batch_size=500, ctx=mx.cpu(),
                 dtype=dtype, act=act) # 'relu'
    return m

def get_trained_model(dim_latent=parameter_util.DIM_LATENT, likelihood_type=parameter_util.SVGP, dataset=None, config={}, load_params=True):
    vae_param_path = parameter_util.VAE_PARAM_PATH
    
    depth = config.get('DEPTH', parameter_util.DEPTH)
    print_iter = config.get('PRINT_ITER', parameter_util.PRINT_ITER)
    n_train = config.get('N_TRAIN', parameter_util.N_TRAIN)
    step_rate = config.get('STEP_RATE', parameter_util.STEP_RATE)
    
    file_name = vae_param_path + dataset.name + '_{}_dims_{}_{}_depth_{}' + '.params'
    
    #Check the parameter files:
    
    numpy_data = dataset.get_data()
    
    params_file_encoder1 = file_name.format('encoder1', numpy_data.shape[-1], dim_latent, depth)
    params_file_encoder2 = file_name.format('encoder2', numpy_data.shape[-1], dim_latent, depth)
    params_file_decoder = file_name.format('decoder', dim_latent, numpy_data.shape[-1], depth)
    #get untrained_model:
    m = get_untrained_model(dim_latent=dim_latent, likelihood_type=likelihood_type, dataset=dataset, config=config)
    
    #Load parameters if they exist
    needs_to_be_optimised = False
    if os.path.isfile(params_file_encoder1) and load_params:
        if print_iter:
            print("Param file {} exists!".format(params_file_encoder1))
        m._comp_logL.encoder1.load_parameters(params_file_encoder1)
        m._comp_logL.encoder2.load_parameters(params_file_encoder2)
        m._comp_logL.decoder.load_parameters(params_file_decoder)
    else:
        if print_iter:
            print("Param file {} does not exists!".format(params_file_encoder1))
        needs_to_be_optimised = True
         
    if needs_to_be_optimised:
        if print_iter:
            print("Optimizing VAE with all MNIST data and saving parameters")
        
        x_cost_reconst, kl_cost = get_model_costs(dataset)[:2]
        
        m.optimize(max_iters=n_train, step_rate=step_rate, y_cost_reconst=None, y_cost_encoder=None, x_cost_reconst=x_cost_reconst, kl_cost=kl_cost, print_iter=print_iter, freeze_vae=False, freeze_likelihood=True, all_data=True, annealing=True)
        
        if print_iter:
            print("Param file {} did not exist before, so we are saving it!".format(params_file_encoder1))
        if load_params:
            m._comp_logL.encoder1.save_parameters(params_file_encoder1)
            m._comp_logL.encoder2.save_parameters(params_file_encoder2)
            m._comp_logL.decoder.save_parameters(params_file_decoder)
    
    else:
        if print_iter:
            print("VAE exists, no need to optimize")
    return m


def get_dataset(dataset_id, config):
    
    if dataset_id == parameter_util.AIRLINE:
        return datasets.AirlineDataset(config)
    
    if dataset_id == parameter_util.SHAPE:
        return datasets.ShapeDataset(config)

    if dataset_id == parameter_util.SMILES:
        return datasets.SMILESDataset(config)
   
    raise ValueError("Unknown dataset " + dataset_id)


def sample(N, lims=None, data=None):    

    if data is not None:
        replace=False
        if N>data.shape[0]:
            replace = True
        ind_init = np.random.choice(data.shape[0], N, replace=replace)
        x_init = data[ind_init,:]
    else:
        x_init = np.random.uniform(size=(N, len(lims)))*(lims[:,1]-lims[:,0]) + lims[:,0]
        x_init = m.predict(mx.nd.array(x_init, dtype=DTYPE))[2]
    return x_init

class ContextForPrint(object):
    """Initialize context environment and replace variables when completed"""
    def __init__(self, redirect=os.devnull):
        self.stdout_old = sys.stdout
        sys.stdout = open(redirect, "w")

    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception_value, traceback):
        sys.stdout = self.stdout_old



