import numpy as np
import mxnet as mx
from utils import run_util
from model import mlp
from utils import parameter_util
from datasets import zinc_grammar
from functools import partial

import itertools

from os import path

from model import costs
import GPy
import pickle

import warnings

import networkx as nx
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
import sascorer

from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
from rdkit.Chem import MolFromSmiles, MolToSmiles


        
from itertools import compress, product, chain




class Dataset():
    def __init__(self, config):
        self.config = config
        self.dtype = self.config.get('DTYPE', parameter_util.DTYPE)
        self.data_dimensionality = parameter_util.DATA_1D
        
    def _get_data(self):
        raise NotImplementedError
    
    def get_data(self):
        return self._get_data()

    def get_data_pure(self):
        return self.get_data()

    def get_x_cost_reconst(self):
        raise NotImplementedError
    
    def get_objective(self):
        raise NotImplementedError

    def get_decoder(self, decoder_units, act=parameter_util.ACT, dtype=parameter_util.DTYPE):
        raise NotImplementedError
    
    def is_binary(self):
        if self.data_type == parameter_util.BINARY:
            return True
        return False   

class ShapeDataset(Dataset):
    def __init__(self, config):
        super(ShapeDataset, self).__init__(config)
        self.name = parameter_util.SHAPE
        self.data_dimensionality = parameter_util.DATA_1D
        
    def get_data(self):
        X = np.genfromtxt('./datasets/data/rectangles10.csv', delimiter=',', dtype=np.float)
        return X
    
    def get_objective(self):
        return self._cost_function
    
    def _cost_function(self, x, model_opt=None, model_ref=None): #Compute the value of the sample
        D = x.shape[-1]
        if (model_opt is not None) and (D == model_opt.d_latent):
            x = model_opt.predict(mx.nd.array(x, dtype=self.dtype), diag=True)[2]
        if type(x) == mx.nd.NDArray:
            x = x.asnumpy()
        return (-np.sum(x, axis=1).reshape((-1,1)) + 25.0)/5.466196599041422
        
    @property
    def data_type(self):
        return parameter_util.BINARY
 
    def get_x_cost_reconst(self):
        return cost_reconst_binary
    
    def get_decoder(self, decoder_units, act=parameter_util.ACT, dtype=parameter_util.DTYPE):
        return mlp.MLP_decoder(decoder_units, act=act, binary=True, prefix='decoder', dtype=dtype)

class MixedDataset(Dataset):
    def __init__(self, config, binary=[], continuous=[], categorical=[], continuousbinary=[], name=None, file_path=None, augment_file_path=None):
        super(MixedDataset, self).__init__(config)
        self.binary = binary
        self.continuous = continuous
        self.categorical = categorical
        self.continuousbinary = continuousbinary
        self.name = name
        self.file_path = file_path
        self.augment_file_path = augment_file_path
        self.shuffle_inds = None
        if augment_file_path is not None and not path.exists(augment_file_path):
            self._augment_data()
    
    def get_x_cost_reconst(self):
        return partial(cost_reconst_mixed, binary=self.binary, categorical=self.categorical, continuous=self.continuous, continuousbinary=self.continuousbinary)
    
    def get_decoder(self, decoder_units, act=parameter_util.ACT, dtype=parameter_util.DTYPE):
        return mlp.MLP_decoder_mixed_data(decoder_units, act=act, prefix='decoder', dtype=dtype, binary=self.binary+self.continuousbinary, categorical=self.categorical, continuous=self.continuous)
    
    def get_data_pure(self):
        return self.get_data()

    def get_objective(self):
        #self._augment_data()
        dataset = get_dataset(self.name, self.config)
        dim_black_box = self.config.get('DIM_BLACK_BOX', parameter_util.DIM_BLACK_BOX)
        m = model_util.get_trained_model(dim_latent=dim_black_box, dataset=dataset, config=self.config, load_params=True)
        objective = partial(self._cost_function, model_ref=m)
        return objective
    
    def _augment_data(self, N=2500):
        augment_file_path = self.augment_file_path
        self.augment_file_path = None
        latent = self.get_data_pure()
        #dataset = get_dataset(self.name, self.config)
        m = model_util.get_trained_model(dim_latent=latent.shape[1], dataset=self, config=self.config)
        X = np.genfromtxt(self.file_path, delimiter=',', dtype=float)[:,:-1]
        m1, s1 = m.predict(mx.nd.array(X, dtype=self.dtype), diag=True)[3:]
        inds = np.random.choice(range(m1.shape[0]), N, replace=True)
        m_, s_ = m1[inds,:], s1[inds,:]
        latent_sample = m_ + np.random.normal(0,1, m_.shape)*s_ 
        X_augment = m.predict(mx.nd.array(latent_sample, dtype=self.dtype))[2]
        #import pdb; pdb.set_trace()
        np.savetxt(augment_file_path, X_augment, delimiter=",")
        self.augment_file_path = augment_file_path

    def _cost_function(self, x, model_opt=None, model_ref=None):
        D = x.shape[-1]
        
        X = mx.nd.array(np.genfromtxt(self.file_path, delimiter=',')[:,:-1], dtype=self.dtype)
        y = np.genfromtxt(self.file_path, delimiter=',')[:,-1].reshape(-1)
        ymax, ymin = np.max(y), np.min(y)
        y = - ( (y - ymin) / (ymax-ymin) ) + 1.0
        y = y / np.std(y)  
        
        if (model_opt is not None) and (D == model_opt.d_latent):
            x = model_opt.predict(mx.nd.array(x, dtype=self.dtype), diag=True)[2]
        
        #Project
        X_ = model_ref.predict(X)[3]
        x_ = model_ref.predict(mx.nd.array(x, dtype=self.dtype), diag=True)[3]
        dists = np.sum((x_[:, None, :] - X_[None,:,:]) ** 2, axis=-1)
        inds = np.argsort(dists, axis=1)[:,:5]
        dists_ = np.empty((0,inds.shape[1]))
        for i in range(inds.shape[0]):
            dists_ = np.r_[dists_, dists[i:i+1, inds[i,:]]]
        
        inv_dist = 1./np.clip(dists_,1e-11,None)
        y_scaled = inv_dist*y[inds]/np.sum(inv_dist,axis=1, keepdims=True)
        return np.mean(y_scaled, axis=1).reshape((-1,1)) #y[inds[:,0]].reshape((-1,1))        


class SMILESDataset(MixedDataset):
    def __init__(self, config, len_grammar=10, len_smiles=21):# len_grammar=34, len_smiles=57 ):
        self.len_grammar = len_grammar
        self.len_smiles = len_smiles
        categorical = [[j*len_grammar+i for i in range(len_grammar)] for j in range(len_smiles)]
        self.file_one_hot  = './datasets/data/SMILES_one_hot.csv'
        super(SMILESDataset, self).__init__(config, categorical=categorical, name=parameter_util.SMILES, file_path='./datasets/data/gdb11_filtered2.smi')
        self.SA_mean, self.logP_mean, self.cycle_mean, self.score_max = (-3.1338599332965997, 0.4198443094629156, -0.005115089514066497, 5.2567988323212855)
        self.SA_std, self.logP_std, self.cycle_std, self.score_std = (0.7683966924165994, 0.9306585573234043, 0.071336704250544, 1.8110317488120244) 
        self.SA_min, self.logP_min, self.cycle_min = (-7.3284153846153846, -2.0180999999999996, -1) 
        self._generate_one_hot_file()
    
    def _generate_one_hot_file(self):
        if not path.exists(self.file_one_hot):
            f = open(self.file_path,'r')
            L = []
            count = -1
            for line in f:
                line = line.strip()
                L.append(line)
            f.close()
            one_hot = zinc_grammar.to_one_hot(L, self.len_smiles, self.len_grammar)
            #import pdb; pdb.set_trace()
            one_hot = one_hot.reshape((one_hot.shape[0], -1))
            
            #smiles = zinc_grammar.to_smiles(one_hot.reshape(-1, self.len_smiles, self.len_grammar))
            
            
            scores = self._cost_function(one_hot, update_refs = True)
            
            #import pdb; pdb.set_trace()
            
            self.score_max = np.max(-scores)
            self.score_std = np.std(scores)
            #import pdb; pdb.set_trace()
            np.savetxt(self.file_one_hot, one_hot, delimiter=",", fmt = "%d")
            

    def get_objective(self):
        return self._cost_function

    def _cost_function(self, x, model_opt=None, model_ref=None, predict_within_data=False, update_refs=False):
        D = x.shape[-1]
        
        if (model_opt is not None) and (D == model_opt.d_latent):
            x = model_opt.predict(mx.nd.array(x, dtype=self.dtype), diag=True)[2]
            
        if not (type(x) is np.ndarray):
            x = x.asnumpy()
        smiles = zinc_grammar.to_smiles(x.reshape(-1, self.len_smiles, self.len_grammar))
        
        logP_values = []
        SA_scores = []
        cycle_scores = []
        
        for i in range(len(smiles)):
            try:
                mol_i = MolFromSmiles(MolToSmiles(MolFromSmiles(smiles[ i ])))
                logP_values.append(Descriptors.MolLogP(mol_i))
                try:
                    SA_scores.append(-sascorer.calculateScore(mol_i))
                except:
                    print("invalid sasscore")
                    SA_scores.append(self.SA_min)
                cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol_i)))
                if len(cycle_list) == 0:
                    cycle_length = 0
                else:
                    cycle_length = max([ len(j) for j in cycle_list ])
                if cycle_length <= 6:
                    cycle_length = 0
                else:
                    cycle_length = cycle_length - 6
                cycle_scores.append(-cycle_length)
                #print("valid smiles: " + smiles[i])
            except:
                #print("Invalid smiles: " + smiles[i])
                if len(logP_values)<i+1:
                    logP_values.append(self.logP_min)
                if len(SA_scores)<i+1:
                    SA_scores.append(self.SA_min)
                if len(cycle_scores)<i+1:
                    cycle_scores.append(self.cycle_min)
                    
        SA_scores = np.array(SA_scores)
        logP_values = np.array(logP_values)
        cycle_scores = np.array(cycle_scores)
        #import pdb; pdb.set_trace()
        if update_refs:
            self.SA_mean = np.mean(SA_scores)
            self.SA_std = np.std(SA_scores)
            self.SA_min = np.min(SA_scores)
            self.logP_mean = np.mean(logP_values)
            self.logP_std = np.std(logP_values)
            self.logP_min = np.min(logP_values)
            self.cycle_mean = np.mean(cycle_scores)
            self.cycle_std = np.std(cycle_scores)
            self.cycle_min = np.min(cycle_scores)

        SA_scores_normalized = (SA_scores - self.SA_mean) / self.SA_std
        logP_values_normalized = (logP_values - self.logP_mean) / self.logP_std
        if self.cycle_std > 0.0:
            cycle_scores_normalized = (cycle_scores - self.cycle_mean) / (self.cycle_std)
        else:
            cycle_scores_normalized = (cycle_scores - self.cycle_mean)
        #print((-(SA_scores_normalized + logP_values_normalized + cycle_scores_normalized - self.score_max)/self.score_std).reshape((-1,1)))
        return (-(SA_scores_normalized + logP_values_normalized + cycle_scores_normalized - self.score_max)/self.score_std).reshape((-1,1))
        
    def get_data(self):
        return np.genfromtxt(self.file_one_hot, delimiter=',', dtype=int)


 

class AirlineDataset(MixedDataset):
    def __init__(self, config, len_kern=4, len_op=3, perc_data=0.66):
        self.len_kern=len_kern
        self.len_op=len_op
        self.num_op = 7
        self.ind_terminating_op = 2
        self.perc_data = perc_data
        categorical = [[i for i in range(len_kern)]]
        for i in range(int((self.num_op-1)/2)):
            categorical += [[j +i*(len_kern+len_op) + len_kern for j in range(len_op)]]
            categorical += [[j +(i+1)*(len_kern+len_op) for j in range(len_kern)]]
        self.len_grammar = categorical[-1][-1]+1
        continuous = [] 
        binary = []
        self.log_lik = None
        self.pre_computed_y = './datasets/data/airline_costs7.csv'
        super(AirlineDataset, self).__init__(config, continuous=continuous, binary=binary, categorical=categorical, name=parameter_util.AIRLINE, file_path='./datasets/data/airline7.csv')

    def _generate_all_grammar_examples(self):
        kerns = []
        for i in range(self.len_kern):
            kern_i = [0]*self.len_kern
            kern_i[i] = 1
            kerns += [kern_i]
        
        ops = []
        for i in range(self.len_op):
            op_i = [0]*self.len_op
            op_i[i] = 1
            ops += [op_i]
        
        combs = [kerns] + [ ops, kerns]*int((self.num_op-1)/2)

        X = np.empty((0, self.len_kern + (self.len_kern+self.len_op)*int((self.num_op-1)/2)), dtype=int)
        for mask in product(*combs):
            X = np.r_[X, np.array([*chain(*mask)]).reshape((1,-1))]
        return X

    
    def _generate_data(self):
        X = self._generate_all_grammar_examples() 
        block_size = self.len_kern+self.len_op
        X_clean = np.zeros_like(X)
        for i in range(X.shape[0]):
            l = X.shape[1]
            for j in range(int((self.num_op)/2)):
                if np.argmax(X[i, j*block_size+self.len_kern:(j+1)*block_size]) == self.ind_terminating_op :
                    l = (j+1)*block_size
                    break
            X_clean[i,:l] = X[i,:l]
        X_clean = np.unique(X_clean, axis=0)
        np.savetxt(self.file_path, X_clean, delimiter=',', fmt='%d')
            


    def get_data(self, gen_file=True):
        if not path.exists(self.file_path):
            self._generate_data()
        return np.genfromtxt(self.file_path, delimiter=',', dtype=int)

    def get_objective(self):
        return self._cost_function

    def _get_bic_and_log_pred(self, codes, X_train, Y_train, X_test=None, Y_test=None, optimize_restarts=10):
        codes = np.atleast_2d(codes)
        codes = codes[:,:self.len_grammar]
        
        n = codes.shape[0]
        bic = np.zeros((n,1))
        l_p = np.zeros((n,1))
        for i in range(n):
            code = codes[i,:]
            #active_dims = code_recon[code_len:-len_distance]
            new_kernel = eval_kernel(kern_params(),code)
            
            m = GPy.models.GPRegression(X_train,Y_train, new_kernel)
            m.optimize(optimizer='bfgs', messages=0, max_iters=1000)
            m.optimize_restarts(optimize_restarts)
            bic[i] = -2*m.log_likelihood()+np.log(len(X_train))*len(new_kernel[:])
            if X_test is not None:
                l_p[i] = np.sum(m.log_predictive_density(X_test, Y_test))/X_test.shape[0]
        self.log_lik = l_p.copy()
        return bic, l_p

    def _cost_function(self, x, model_opt=None, optimize_restarts=10):
        D = x.shape[-1]
        if (model_opt is not None) and (D == model_opt.d_latent):
            x = model_opt.predict(mx.nd.array(x, dtype=self.dtype), diag=True)[2]
        if not (type(x) is np.ndarray):
            x = x.asnumpy()
        
        x = np.array(x, dtype=int)
        
        codes = self.get_data()
        
        block_size = self.len_kern+self.len_op
        x_clean = np.zeros_like(x, dtype=int) 
        inds = []
        for i in range(x.shape[0]):
            l = x.shape[1]
            for j in range(int((self.num_op)/2)):
                if np.argmax(x[i, j*block_size+self.len_kern:(j+1)*block_size]) == self.ind_terminating_op :
                    l = (j+1)*block_size
                    break
            x_clean[i,:l] = x[i,:l]
            inds += [np.argmax(1 - np.any(codes - x_clean[i:i+1,:], axis=1))]
        
        costs = np.genfromtxt('./datasets/data/airline_costs7.csv', delimiter=',')[inds,:]
        
        bic = costs[:,0:1]
        
        self.log_lik = costs[:,1:2]
        
        return bic        
        

def kern_params():
    param_constrains = (10e-6,100)
    k1 = GPy.kern.RBF(input_dim=1)
    k1.variance = np.exp(np.random.normal(loc=0.4, scale=0.7, size=None))
    k1.lengthscale = np.exp(np.random.normal(loc=0.1, scale=0.7, size=None))
    k1.variance.constrain_bounded(*param_constrains)
    k1.lengthscale.constrain_bounded(*param_constrains)

    k2 = GPy.kern.Add([GPy.kern.Linear(input_dim=1),GPy.kern.Bias(input_dim=1)])
    k2.linear.variances = np.exp(np.random.normal(loc=0.4, scale=0.7, size=None))
    k2.bias.variance  = np.random.normal(loc=0, scale=2, size=None)
    k2.linear.variances.constrain_bounded(*param_constrains)
    k2.bias.variance.constrain_bounded(*(10e-6,10))

    k3 = GPy.kern.StdPeriodic(input_dim=1)
    k3.variance = np.exp(np.random.normal(loc=0.4, scale=0.7, size=None))
    k3.period = np.exp(np.random.normal(loc=0.1, scale=0.7, size=None))
    k3.lengthscale = np.exp(np.random.normal(loc=2, scale=0.7, size=None))
    k3.variance.constrain_bounded(*param_constrains)
    k3.period.constrain_bounded(*param_constrains)
    k3.lengthscale.constrain_bounded(*param_constrains)

    k4 = GPy.kern.RatQuad(input_dim=1)
    k4.variance = np.exp(np.random.normal(loc=0.4, scale=0.7, size=None))
    k4.lengthscale = np.exp(np.random.normal(loc=0.1, scale=0.7, size=None))
    k4.power = np.exp(np.random.normal(loc=0.05, scale=0.7, size=None))
    k4.variance.constrain_bounded(*param_constrains)
    k4.lengthscale.constrain_bounded(*param_constrains)
    k4.power.constrain_bounded(*param_constrains)
    return [k1,k2,k3,k4]

def Code2Grammar(code,kern=False, kern_dict=None, op_dict=None):
    
    len_kern = len(kern_dict)
    len_op = len(op_dict)
    code_len = int((len(code)-len_kern)/(len_kern+len_op)*2+1)
    sentence = []
    add_index = 0

    for iter in range(code_len):
        if (iter % 2 == 0 ):
            if kern:
                sentence.extend(kern_dict[np.nonzero(code[range(int(iter/2*(len_kern+len_op)),int((iter+2)/2*len_kern + iter/2*len_op))])[0][0]] + '.copy()')     
            else:
                sentence.extend(kern_dict[np.nonzero(code[range(int(iter/2*(len_kern+len_op)),int((iter+2)/2*len_kern + iter/2*len_op))])[0][0]] )
        else:
            op_index = np.nonzero(code[range(int((iter+1)/2*(len_kern+len_op)-len_op),int( (iter+1)/2*(len_kern+len_op)))])[0][0]
            if op_index == 2:
                break
            else:
                sentence.extend(op_dict[op_index])
    kernel =  ''.join(sentence)
    print(kernel)
    return kernel

def eval_kernel(kern_params,code):
    param_constrains = (10e-6,10e6)
    
    kern_dict = ['k1','k2','k3','k4'] # self.kern_dict
            
    for kk in range(len(kern_dict)):
        exec(kern_dict[kk] + " = kern_params[kk]")
    #import pdb; pdb.set_trace()
    new_kernel = eval(Code2Grammar(code, kern=True, kern_dict=kern_dict, op_dict=['+','*',None]))

    return new_kernel

def cost_reconst_binary(F, X, X_tilde, dtype, soft_zero=1e-7, annealing=1.0):
    X_tilde = F.clip(X_tilde, soft_zero, 1-soft_zero)
    return F.reshape(F.sum(X*F.log(X_tilde) + (1-X) * F.log(1-X_tilde), axis=1), shape=-1)

def cost_reconst_categorical(F, X, X_tilde, dtype, soft_zero=1e-10, annealing=1.0):
    return F.reshape(F.sum(X*F.log(X_tilde), axis=1), shape=-1) 
    

def cost_reconst_continuous_binary(F, X, X_tilde, dtype, soft_zero=1e-5, annealing=1.0):
    X = F.flatten(X)
    X_tilde = F.flatten(X_tilde)
    X_tilde = F.clip(X_tilde, soft_zero, 1-soft_zero)
    C = stableC(F, X_tilde, soft_zero = soft_zero)
    return F.reshape(F.sum(X*F.log(X_tilde) + (1-X) * F.log(1-X_tilde) + annealing*F.log(C), axis=1), shape=-1) 

def cost_reconst_continuous(F, X, X_tilde, dtype, soft_zero=1e-7, annealing=1.0):
    X_tilde_mean, X_tilde_sigma = X_tilde
    X_tilde_sigma = X_tilde_sigma+soft_zero

    return F.reshape(F.sum(-F.log(X_tilde_sigma) - 0.5*F.square((X-X_tilde_mean)/X_tilde_sigma) - np.log(2*np.pi)/2, axis=1), shape=-1) 

def cost_reconst_mixed(F, X, X_tilde, dtype, binary=[], categorical=[], continuous=[], continuousbinary=[], soft_zero=1e-7, annealing=1.0):
    X_tilde_mean, X_tilde_sigma = X_tilde
  
    cost_binary = cost_reconst_binary(F, X[:,binary], X_tilde_mean[:,binary], dtype, soft_zero=soft_zero, annealing=annealing) if len(binary)>0 else 0
    
    all_categorical = list(itertools.chain(*categorical))
    cost_categorical = cost_reconst_categorical(F, X[:,all_categorical], X_tilde_mean[:,all_categorical], dtype, soft_zero=soft_zero, annealing=annealing) if len(all_categorical)>0 else 0.0
    
    cost_continuous = annealing*cost_reconst_continuous(F, X[:,continuous], (X_tilde_mean[:,continuous], X_tilde_sigma), dtype, annealing=annealing) if len(continuous)>0 else 0
    cost_continuousbinary = cost_reconst_continuous_binary(F, X[:,continuousbinary], X_tilde_mean[:,continuousbinary], dtype, soft_zero=soft_zero, annealing=annealing) if len(continuousbinary)>0 else 0
    return cost_binary + cost_categorical + cost_continuous + cost_continuousbinary

def stableC(F, x, soft_zero = 1e-5):
    '''
    Numerically stable implementation of 
    Continous Bernoulli constant C,
    using Taylor 2nd degree approximation
 
    ''' 
    #import pdb; pdb.set_trace()
    
    mask = x > 0.5
    x  = F.clip( F.abs(x-0.5), soft_zero, 0.5-soft_zero)
    
    x = F.where(mask, 0.5 + x, 0.5 - x)
    
    mask2 = F.abs(x- 0.5) >= soft_zero    
    C  = F.where(mask2, 
                 (F.log(1. - x) - F.log(x))/(1. - 2.*x),
                 2 + F.log(1. + 1./3. * (1. - 2. * x)**2))
    
    return C