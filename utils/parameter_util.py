import os
import numpy as np


#All possible Datasets
AIRLINE = 'airline'
SHAPE = 'shape'
SMILES = 'smiles'

#Data types (The data in each dataset is a combination of one or more of these)
BINARY = 0
CONTINUOUSBINARY = 1
CONTINUOUS = 2

#DATA D:
DATA_1D = 0
DATA_IMAGE = 1

#GP types
SVGP = 0

#Acquisition functions
RANDOM = 0
TS = 1
QEI = 2
LCB = 3
PI = 4

#Space optimization:
NO  = 0
ELLIPS = 1
CONVEX = 2
STABLE = 3

RES_PATH = 'results/' 
VAE_PARAM_PATH = RES_PATH + 'vae-params/'

if not os.path.exists(RES_PATH):
    os.makedirs(RES_PATH)
if not os.path.exists(VAE_PARAM_PATH):
    os.makedirs(VAE_PARAM_PATH)



#Define different methods used in the paper
METHODS = {'baseline': {'ALL_DATA': False,  'FREEZE_VAE':True, 'OPTIMIZED_SPACE':NO},
           'jointbaseline': {}}

#method defaults:
FREEZE_VAE = False #If VAE is frozen during BO
ALL_DATA = True # Use all data or only the labelled observations
OPTIMIZED_SPACE = NO # What acquisition optimization space restriction method is used

#other defaults
DTYPE = np.float64
NUM_INDUCING = 150 # Number of inducing points
NUM_SAMPLES_H = 1 # Number of samples used in VAE latent space to represent the uncertianty of the latent space
PRINT_ITER = True # Print optimization result
FIX_SIGMA = None # Fix variance of the likelihood: None=optimize from data, e.g. 1e-2= fix to 1e-2
ACT = 'relu' # activation function of the NNs
DEPTH = 4 # depth of encoder and decoder
LENGTHSCALE = 1 # Initial value for lengtscale
VARIANCE = 1 # initial value for kernel lengthscale
ARD = True # each dimension has own lengthscale
FIX_KERN = False # fix kernel parameters of GP
DIM_LATENT = 2 # dimensionality of the latent space
ACQ = QEI # acquisition function
METHOD = 'baseline' # gp vae optimization method
SEED = 0 # random seed
N_ITER = 100 # number of iterations the BO is ran
INIT_SIZE = 50 # number of initail samples
DATASET = SHAPE # Dataset to be used
PLOT = False # plot the latent space
OPTIMIZE_RATE  = 10 # how often the latent space is modified (parameters of the VAE are changed)
OPTIMIZED_SPACE = NO # optimization restriction strategy


STEP_RATE = 1e-3 # Step rate used in optimization
N_INIT = 700 # number of epochs when training the VAE without observations
N_LOOP = 500 # number of epochs during BO
N_TRAIN = 1000 # Number of epochs for saving the parameters

#DIMS tests in the paper
ALTERNATIVES = [('DATASET', [SHAPE, SMILES, AIRLINE]), 
                ('SEED', [i for i in range(10)]),
                ('DIM_LATENT', [2,3,4,5,6,7]),
                ('METHOD', ['baseline', 'jointbaseline']), 
                ('OPTIMIZED_SPACE', [NO]),
                ('ACQ', [LCB])] 

#OPTIMIZED_SPACE tests in the paper
ALTERNATIVES = [('DATASET', [SHAPE, SMILES, AIRLINE]), 
                ('SEED', [i for i in range(10)]),
                ('DIM_LATENT', [4]),
                ('METHOD', ['baseline', 'jointbaseline']), 
                ('OPTIMIZED_SPACE', [NO, CONVEX, ELLIPS, STABLE]),
                ('ACQ', [LCB])] 


#ACQ tests in the paper
ALTERNATIVES = [('DATASET', [SHAPE, SMILES, AIRLINE]),
                ('SEED', [i for i in range(10)]),
                ('DIM_LATENT', [4]),
                ('METHOD', ['baseline', 'jointbaseline']), 
                ('OPTIMIZED_SPACE', [NO]),
                ('ACQ', [LCB, QEI, PI, TS])] 


N_ITER = 1
ALTERNATIVES = [('DATASET', [SHAPE]),
                ('SEED', [i for i in range(1)]),
                ('DIM_LATENT', [4]),
                ('METHOD', ['baseline']), 
                ('OPTIMIZED_SPACE', [CONVEX, ELLIPS, STABLE]),
                ('ACQ', [RANDOM])] 

#ENUM names to be printed:
ENUMS = {'ACQ':{TS:"Thompson Sampling", QEI:"Expected Improvement", RANDOM:"Random", LCB:"Lower Confidence Bound", PI:"Probability of Improvement"}, 'OPTIMIZED_SPACE':{NO:"Hypercube", ELLIPS:"Hyperellipsoid", CONVEX:'Convex hull', STABLE:"Distance"},'DATASET':{AIRLINE:'Airline', SMILES:'SMILES', SHAPE:'Shape'}, 'METHOD':{'jointbaseline':'joint training', 'baseline':'disjoint training'}}
NAME_ENUMS = {'OPTIMIZED_SPACE':"Optimization space", 'ACQ':'Acquisition function', 'DIM_LATENT':'Latent space dimensionality'}
WANTED_ORDERS = {}

