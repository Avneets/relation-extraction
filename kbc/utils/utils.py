import sys
from collections import OrderedDict
import six.moves.cPickle as pickle

import numpy as np


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if minibatch_start != n:
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def get_params(model):
    model_all_params = {}
    for param_name, model_param in model.all_params_dict.items():
        model_all_params[param_name] = model_param.get_value()
    return model_all_params


def save(saveto, all_params):
    print('Saving params...')

    pickle.dump(all_params, open('%s.pkl' % saveto, 'wb'), -1)
    print('Done')


def load_params(savedfile):
    try:
        model_all_params = pickle.load(open('%s.pkl' % savedfile, 'rb'))
        return model_all_params
    except:
        print("Couldn't load params. Fresh initialization? [y/n]")
        inp = raw_input()
        if inp[0] == 'y': return None
        else: sys.exit(1)
        return None


