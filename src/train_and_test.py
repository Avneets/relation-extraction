#
# Some methods taken from
# http://deeplearning.net/tutorial/code/lstm.py
#

# notes:
# change saveTo to config

from __future__ import print_function

import theano
import theano.tensor as T
import numpy as np
import models
from collections import OrderedDict
import config
from reader import Reader
import logging
import time
import six.moves.cPickle as pickle
import sys


# Set the random number generators' seeds for consistency
SEED = 123
np.random.seed(SEED)


def numpy_floatX(data):
    return np.asarray(data, dtype=theano.config.floatX)


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


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def _p(pp, name):
    return '%s_%s' % (pp, name)


def load_params(path, params):
    pp = np.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def adadelta(lr, tparams, grads, x_triples, x_subj_neg, x_obj_neg, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tparams: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x_triples: Theano variable
        Model input triples in the form of {(e_s, r, e_o)}_i matrix
    x_subj_neg: Theano variable
        Model negative samples replacing subject position
    x_obj_neg: Theano variable
        Model negative samples replacing object position
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x_triples, x_subj_neg, x_obj_neg], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x_triples, x_subj_neg, x_obj_neg, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tparams: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x_triples: Theano variable
        Model input triples in the form of {(e_s, r, e_o)}_i matrix
    x_subj_neg: Theano variable
        Model negative samples replacing subject position
    x_obj_neg: Theano variable
        Model negative samples replacing object position
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x_triples, x_subj_neg, x_obj_neg], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / T.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def mrr(ranks):
    if "float" not in str(ranks.dtype):
        ranks = numpy_floatX(ranks)

    return np.mean(1 / ranks)


def hits_at_k(ranks, k):
    if "float" not in str(ranks.dtype):
        ranks = numpy_floatX(ranks)

    return float(np.nonzero(ranks <= k)[0].shape[0]) / float(ranks.shape[0])


def train_model(
        param_names,            # list of parameter names - note that the order is important
        dim_emb=10,                 # Dimensionality of the embeddings of entities and relations
        num_neg=100,   # number of negative samples for each positive instance triple
        optimizer=rmsprop,              # write down the optimizers available in this module
        model_file="model.npz",     # model file which stores the params
        reload_model=False,          # whether to reload the model params from model_file
        L1_reg=0.0,                 # L1 regularisation coefficient
        L2_reg=1.0,                  # L2 regularisation coefficient
        max_epochs=100,              # max number of epochs to train
        batch_size=128,
        lrate=0.001,
        dispFreq=10,
        saveFreq=1110,
        validFreq=370,
        saveto='distmult_model.npz',
        patience=10,
        full_train=True,
        num_train=1000
):
    model_options = locals().copy()  # has all the parameters required for the model
    print("model options", model_options)

    logging.info("loading train KB data")
    train_dataset = Reader.read_data(config.KBTrainFile)
    train_dataset.print_set_statistics()
    logging.info("loading validation KB data")
    valid_dataset = Reader.read_data(filepath=config.KBValidationFile,
                                     entity_dict=train_dataset.entity_index,
                                     relation_dict=train_dataset.relation_index,
                                     add_new=False
                                     )
    valid_dataset.print_set_statistics()
    logging.info("loading test KB data")
    test_dataset = Reader.read_data(filepath=config.KBTestFile,
                                    entity_dict=train_dataset.entity_index,
                                    relation_dict=train_dataset.relation_index,
                                    add_new=False
                                    )
    test_dataset.print_set_statistics()
    logging.info("generating full train data with negative samples")
    if full_train:
        batch_data = train_dataset.generate_batch(full_data=True)
    else:
        batch_data = train_dataset.generate_batch(batch_size=num_train)

    logging.info("building model")

    # load the embeddings from file - add the functionality
    params = OrderedDict()
    for name in param_names:
        params[name] = None
    if reload_model:
        load_params(model_file, params)
    model = models.DistMult(n_entities=train_dataset.n_entities,
                            n_relations=train_dataset.n_relations,
                            n_dim=dim_emb,
                            L1_reg=L1_reg,
                            L2_reg=L2_reg,
                            params=params)

    tparams = model.get_tparams()

    x_triples = T.imatrix()             # input triples
    x_subj_neg = T.imatrix()            # in_sub_neg_samples_mat
    x_obj_neg = T.imatrix()             # in_obj_neg_samples_mat

    cost = model.total_cost(x_triples, x_subj_neg, x_obj_neg)

    grads = T.grad(cost, wrt=list(tparams.values()))
    lr = T.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x_triples, x_subj_neg, x_obj_neg, cost)

    train_triples = batch_data[0].astype(np.int32)
    valid_triples = valid_dataset.generate_batch(full_data=True, no_neg=True)[0].astype(np.int32)
    test_triples = test_dataset.generate_batch(full_data=True, no_neg=True)[0].astype(np.int32)

    pred_ranks = model.pred_ranks(x_triples)
    get_pred_ranks = theano.function(
        inputs=[x_triples],
        outputs=pred_ranks
    )

    history_errs = []
    best_p = None
    bad_counter = 0

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in range(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train_triples), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1

                # change to floatX and T.cast into int for performance
                train_triples_kf = train_triples[train_index]
                train_sub_neg = batch_data[1][train_index].astype(np.int32)
                train_obj_neg = batch_data[2][train_index].astype(np.int32)

                n_samples += len(train_triples_kf)

                cost = f_grad_shared(train_triples_kf, train_sub_neg, train_obj_neg)
                f_update(lrate)

                if np.isnan(cost) or np.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 0., 0., 0., model, None

                if np.mod(uidx, dispFreq) == 0:
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost)

                if saveto and np.mod(uidx, saveFreq) == 0:
                    print('Saving...')

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    np.savez(saveto, history_errs=history_errs, **params)
                    pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print('Done')

                if np.mod(uidx, validFreq) == 0:

                    train_ranks = get_pred_ranks(train_triples_kf)
                    valid_ranks = get_pred_ranks(valid_triples)
                    test_ranks = get_pred_ranks(test_triples)

                    train_err = mrr(train_ranks)
                    valid_err = mrr(valid_ranks)
                    test_err = mrr(test_ranks)

                    history_errs.append([valid_err, test_err])

                    if (best_p is None or
                        valid_err <= np.array(history_errs)[:,
                                                               0].min()):

                        best_p = unzip(tparams)
                        bad_counter = 0

                    print( ('Train ', train_err, 'Valid ', valid_err,
                           'Test ', test_err) )

                    if (len(history_errs) > patience and
                        valid_err >= np.array(history_errs)[:-patience,
                                                               0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print('Early Stop!')
                            estop = True
                            break

            print('Seen %d samples' % n_samples)

            if estop:
                break

    except KeyboardInterrupt:
        print("training interrupted")

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    print('calculating training err')

    # kf = get_minibatches_idx(len(train_triples), batch_size, shuffle=False)
    train_err = mrr(get_pred_ranks(train_triples))
    print('calculating validation err')
    valid_err = mrr(get_pred_ranks(valid_triples))
    print('calculating test err')
    test_err = mrr(get_pred_ranks(test_triples))

    print( 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err )
    if saveto:
        np.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print('The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
    print( ('Training took %.1fs' %
            (end_time - start_time)), file=sys.stderr )
    return train_err, valid_err, test_err, model, None

if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    _, _, _, model, debug = train_model(
        param_names=[config.ENTITY_EMBEDDINGS, config.RELATION_EMBEDDINGS],
        max_epochs=10,
        full_train=False,
        num_train=4096,
        L2_reg=0.1,
        L1_reg=0.01,
        batch_size=512
    )





