import sys

import numpy as np

from kbc.reader import KBReader
from kbc.utils import config, utils, kb_utils
from kbc.models import models
from kbc.utils.params import *


def launch(params):
    # Load the dataset
    print("Loading data...")
    rng = np.random
    print("params", params)
    debug_data = {}

    n_entities, n_relations, train_triples, valid_triples, test_triples, sr_index = kb_utils.load_kb_data(params[TRAIN_FRAC], params[VALID_FRAC], params[TEST_FRAC])

    print("Building model and compiling functions...")

    all_params_init = None
    if params[RELOAD_MODEL]:
        all_params_init = utils.load_params(params[MODEL_FILE])

    if params[MODEL_NAME] == MODEL_E:
        Model = models.Model2
    elif params[MODEL_NAME] == DISTMULT:
        Model = models.Model3
    elif params[MODEL_NAME] == DISTMULT_AND_E:
        Model = models.Model2plus3
    else:
        print("Model %s doesn't exist. Fatal error!")
        sys.exit(1)

    model = Model(n_entities, n_relations, params[DIM_EMB], all_params_init, is_normalized=params[IS_NORMALIZED], L1_reg=params[L1_REG], L2_reg=params[L2_REG])
    return kb_utils.train(model, train_triples, valid_triples, test_triples, sr_index, params)


if __name__ == '__main__':
    local_params = {MODEL_NAME: MODEL_E,
                    MODEL_FILE: 'model2test', SAVETO_FILE: 'model2test', RELOAD_MODEL: True,
                    DISP_FREQ: 2, VALID_FREQ: 10, SAVE_FREQ: 10,
                    NUM_EPOCHS: 50,
                    TRAIN_FRAC: 1.0/270, VALID_FRAC: 0.1, TEST_FRAC: 0.1,
                    L1_REG: 0., L2_REG: 0., IS_NORMALIZED: True,
                    BATCH_SIZE: 128, LEARNING_RATE: 0.01, NUM_NEG: 10
                    }
    p = Params(local_params)
    ranks = launch(params=p)
