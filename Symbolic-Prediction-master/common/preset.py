import numpy as np

# --------------------------
# Data generation parameters
# --------------------------

# total number of symbolic tree instances
NTREE = 20000
# number of (x, y) pairs for each instance
NSAMPLE = 10000
# number of explanatory variables
NVAR = 2
# proportion of training set
TRAIN_PROP = 0.8
# proportion of validation set
VAL_PROP = 0.1

# range of constants
CONSTANT_RANGE = (-1, 1)
# probability of a node being a leaf
TERMINAL_PROB = 0.4
# probability of a leaf node being a constant
CONSTANT_PROB = 0.2
# depth constraints of the symbolic tree
MAX_DEPTH = 4
MIN_DEPTH = 2
# explanatory variable is limited to [VAR_RANGE_LOW, VAR_RANGE_HIGH]
VAR_RANGE_LOW = -10
VAR_RANGE_HIGH = 10
# response data is limited to [-Y_TRESH, Y_THRESH]
Y_THRESH = 100
# standard deviation of the gaussian noise added to the data
NOISE = 0.01


NDIM = 2
import math as _math
NPROJ = _math.factorial(NVAR) // \
	(_math.factorial(NDIM) * _math.factorial(NVAR - NDIM))


# ---------------------
# Data and result paths
# ---------------------

DATA_PATH_TEMPLATE = "data/data_{}.h5"
SUPERR_PATH_TEMPLATE = "data/superr_{}.h5"
SUPERR_ENCODER_WEIGHT_PATH_TEMPLATE = "weights/superr_encoder_{}.h5"
MLP_ENCODER_PATH_TEMPLATE = "data/MLP_encoder_{}.h5"
DECODER_WEIGHT_PATH_TEMPLATE = "weights/decoder_{}_{}.h5"
RESULT_PATH_TEMPLATE = "results/model/result_{}.pkl"
GA_RESULT_PATH_TEMPLATE = "results/SR/result_{}.pkl"
EXPERIMENT_PATH_TEMPLATE = "data/experiment/experiment_{}.h5"
EXPERIMENT_RESULT_PATH_TEMPLATE = "results/experiment/result_{}.pkl"

MLP_BASELINE_PATH_TEMPLATE = "results/baseline/MLP_baseline_{}_{}.pkl"
GA_BASELINE_PATH_TEMPLATE = "results/baseline/GA_baseline_{}.pkl"

DATA_PATH = DATA_PATH_TEMPLATE.format(NVAR)
SUPERR_PATH = SUPERR_PATH_TEMPLATE.format(NVAR)
RESULT_PATH = RESULT_PATH_TEMPLATE.format(NVAR)
GA_RESULT_PATH = GA_RESULT_PATH_TEMPLATE.format(NVAR)
SUPERR_ENCODER_WEIGHT_PATH = SUPERR_ENCODER_WEIGHT_PATH_TEMPLATE.format(NVAR)
MLP_ENCODER_PATH = MLP_ENCODER_PATH_TEMPLATE.format(NVAR)


# ---------------------
# Model hyperparameters
# ---------------------

ENCODING_SCHEME = "super resolution"
BATCH_SIZE = 20

# super-resolution encoder
LOW_RESOLUTION = (50, 50)
HIGH_RESOLUTION = (200, 200)
SUPERR_LR = 1e-4
SUPERR_EPOCHS = 30
SUPERR_PATIENCE = 10
INTERPOLATION_ITER = 20

# MLP encoder
MLP_ENCODER_ALPHA = 1e-3
MLP_ENCODER_LR_INIT = 5e-3
MLP_ENCODER_MAX_ITER = 2
MLP_ENCODER_SIZE = (200, 200, 200,)
MLP_ENCODER_ITER = 10
MLP_ENCODER_THRESHOLD = 0.8
MLP_ENCODER_R2_THRESHOLD = 0.0

# decoder
CNN_REGULARIZATION = 5e-4
DECODER_LR = 2e-4
DECODER_MOMENTUM = 0.9
DECODER_EPOCHS = 50
DECODER_PATIENCE = 10

# attention
ATTENTION_REGULARIZATION = 5e-4
ATTENTION_DIM = 250

# GA
GA_POPULATION = 1000
GA_PARSIMONY = 1e-4


# ---------------
# Baseline models
# ---------------

# MLP baseline
MLP_BASELINE_SIZE = (2048, 2048, 2048, 2048,)
MLP_BASELINE_LR_INIT = 1e-3
MLP_BASELINE_NPERM = 50

# GA baseline
GA_BASELINE_GENERATION = 10
GA_BASELINE_POPULATION = 200


# -----------
# Experiments
# -----------

EXPERIMENT_NSAMPLE = 500000
EXPERIMENT_NOISE = 1e-4
EXPERIMENT_NGEN = 20
EXPERIMENT_NTRIALS = 100