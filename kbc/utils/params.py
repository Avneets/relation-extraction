import os

import config

# Hyper parameters
BATCH_SIZE = 'batch_size'
NUM_EPOCHS = 'num_epochs'
LEARNING_RATE = 'learning_rate'

# Runtime parameters
TRAIN_FRAC = 'train_frac'
TEST_FRAC = 'test_frac'
VALID_FRAC = 'valid_frac'
VALID_FREQ = 'valid_freq'
DISP_FREQ = 'disp_freq'
SAVE_FREQ = 'save_freq'

# Save parameters
MODEL_FILE = 'model_file'
SAVETO_FILE = 'saveto_file'
RELOAD_MODEL = 'reload_model'

# Modelling parameters
DIM_EMB = 'dim_emb'
L1_REG = 'l1_reg'
L2_REG = 'l2_reg'
OTHER_REG = 'other_reg'
NUM_NEG = 'num_neg'
IS_NORMALIZED = 'is_normalized'
MARGE = 'marge'

MODEL_NAME = 'model_name'
# KB models
DISTMULT = 'distmult'
MODEL_E = 'modelE'
DISTMULT_AND_E = 'distmult+E'

params_list = [
    MODEL_NAME,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,         # hyperparams
    TRAIN_FRAC, TEST_FRAC, VALID_FRAC, VALID_FREQ, DISP_FREQ, SAVE_FREQ,             # runtime_params
    MODEL_FILE, SAVETO_FILE, RELOAD_MODEL,                         # save_params
    DIM_EMB, L1_REG, L2_REG, OTHER_REG, NUM_EPOCHS, IS_NORMALIZED, MARGE  # modelling_params
    ]

default_params = {
    MODEL_NAME: MODEL_E,
    BATCH_SIZE: 128,
    NUM_EPOCHS: 100,
    LEARNING_RATE: 0.01,
    TRAIN_FRAC: 1.0,
    TEST_FRAC: 1.0,
    VALID_FRAC: 1.0,
    VALID_FREQ: 1000,
    DISP_FREQ: 100,
    SAVE_FREQ: 5000,
    MODEL_FILE: os.path.join(config.saveModelsRoot, 'model'),
    SAVETO_FILE: os.path.join(config.saveModelsRoot, 'model'),
    RELOAD_MODEL: False,
    DIM_EMB: 10,
    L1_REG: 0.,
    L2_REG: 0.,
    OTHER_REG: 0.,
    NUM_NEG: 10,
    IS_NORMALIZED: True,
    MARGE: 1.0
}


class Params(dict):

    def __init__(self, init):
        if isinstance(init, dict):
            dict.__init__(self, init)
        else:
            print 'Warning: Unknown initialisation argument for Params. Ignoring initialization'
        for kk, vv in default_params.items():
            if kk not in self.keys():
                self[kk] = vv