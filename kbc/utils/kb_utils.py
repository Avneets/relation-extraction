import time
import math
from collections import defaultdict

import numpy as np

import config, utils
from kbc.reader import KBReader
from kbc.utils.params import *


def load_kb_data(train_frac, valid_frac, test_frac):
    train_file = config.KBTrainFile
    valid_file = config.KBValidationFile
    test_file = config.KBTestFile

    train = KBReader.read_data(train_file)
    train.print_set_statistics()

    valid = KBReader.read_data(
        filepath=valid_file,
        entity_dict=train.entity_index,
        entity_pair_dict=train.entity_pair_index,
        relation_dict=train.relation_index,
        add_new=False
    )
    valid.print_set_statistics()

    test = KBReader.read_data(
        filepath=test_file,
        entity_dict=train.entity_index,
        entity_pair_dict=train.entity_pair_index,
        relation_dict=train.relation_index,
        add_new=False
    )
    test.print_set_statistics()

    if train_frac < 1.0:
        train_triples = train.generate_batch(batch_size=int(train.n_triples*train_frac))[0].astype(config.intX)
    else:
        train_triples = train.generate_batch(batch_size=-1)[0].astype(config.intX)

    valid_triples = valid.generate_batch(batch_size=int(valid_frac*valid.n_triples))[0].astype(config.intX)
    test_triples = test.generate_batch(batch_size=int(test_frac*test.n_triples))[0].astype(config.intX)

    sr_index = build_sub_rel_index(train.triples, valid.triples, test.triples)

    return train.n_entities, train.n_relations, train_triples, valid_triples, test_triples, sr_index


def build_sub_rel_index(*triple_sets):
    sr_index_set = defaultdict(set)
    for triple_set in triple_sets:
        for triple in triple_set:
            s, r, o = triple
            sr_index_set[(s, r)].add(o)
    sr_index = dict()
    for k, v in sr_index_set.items():
        sr_index[k] = np.array(list(v))
    return sr_index


def get_ranks(scores, e_os):
    e_os_scores = scores[np.arange(scores.shape[0]), e_os]
    ranks_os = ( scores >= e_os_scores.reshape((-1, 1)) ).sum(axis=1)

    return ranks_os


def get_discounted_ranks(scores, in_triples, sr_index):
    e_os = in_triples[:, 2]
    e_os_scores = scores[np.arange(scores.shape[0]), e_os]
    ranks_os = ( scores >= e_os_scores.reshape((-1, 1)) ).sum(axis=1)

    competing_indices = [ sr_index[tuple(sr_pair)] for sr_pair in in_triples[:, :2] ]
    competing_scores = np.array([ np.sum(scores[index, competing_list] > e_os_scores[index]) for index, competing_list in enumerate(competing_indices) ])

    ranks_os = ranks_os - competing_scores
    return ranks_os


def get_batch_metrics(triples, sr_index, scores_fn, incl_discounting=False):
    bins = [1, 11, 21, 31, 51, 101, 1001, 10001, 20000]

    scores = scores_fn(triples)[0]
    ranks = get_ranks(scores, triples[:, 2])
    mean_rank = np.mean(ranks)
    hits10 = float((ranks <= 10).sum()) * 100. / ranks.shape[0]
    dist = np.histogram(ranks, bins)[0]
    dist_percent = dist.astype(dtype=config.floatX) * (100. / ranks.shape[0])

    print("    mean rank: {:.3f}".format(mean_rank))
    print("    rank dist: {!s}".format(dist))
    print("    rank dist as %: {!s}".format(dist_percent))
    print("    hits@10: {:.3f}".format(hits10))

    if incl_discounting:
        disc_ranks = get_discounted_ranks(scores, triples, sr_index)
        mean_disc_ranks = np.mean(disc_ranks)
        disc_hits10 = float((disc_ranks <= 10).sum()) * 100. / ranks.shape[0]

        print("    mean discounted rank: {:.3f}".format(mean_disc_ranks))
        print("    discounted hits@10: {:.3f}".format(disc_hits10))

        return disc_hits10
    else:
        return hits10


def get_best_metric(history):
    if len(history) > 0:
        best_metric = np.max(history)
        print("Performance = {:.3f}".format(best_metric))
        return best_metric
    else:
        return None


def train(model, train_triples, valid_triples, test_triples, sr_index, params):
    rng = np.random
    n_entities, n_relations = model.n_entities, model.n_relations

    train_fn = model.train_fn(num_neg=params[NUM_NEG], lrate=params[LEARNING_RATE], marge=params[MARGE])
    ranks_fn = model.ranks_fn()
    scores_fn = model.scores_fn()

    uidx = 1
    best_p = None
    history_valid_hits = []
    history_test_hits = []
    history_epoch_times = []
    bins = [1, 11, 21, 31, 51, 101, 1001, 10001, 20000]
    print("Training on {:d} triples".format(len(train_triples)))
    num_batches = int(math.ceil(len(train_triples) / params[BATCH_SIZE]))
    print("Batch size = {:d}, Number of batches = {:d}".format(params[BATCH_SIZE], num_batches))
    print("The eval is being printed with number of items the bins -> %s" % bins)
    try:
        # We iterate over epochs:
        train_start_time = time.time()
        for epoch in range(params[NUM_EPOCHS]):
            # In each epoch, we do a full pass over the training data:
            epoch_start_time = time.time()
            for _, train_index in utils.get_minibatches_idx(len(train_triples), params[BATCH_SIZE], False):
                # Normalize the entity embeddings
                if params[IS_NORMALIZED]:
                    model.normalize()

                tmb = train_triples[train_index]

                # generating negative examples replacing left entity
                tmbln_list = [rng.randint(0, n_entities, tmb.shape[0]).astype(dtype=tmb.dtype) for i in xrange(params[NUM_NEG])]

                # generating negative examples replacing right entity
                tmbrn_list = [rng.randint(0, n_entities, tmb.shape[0]).astype(dtype=tmb.dtype) for i in xrange(params[NUM_NEG])]

                cost = train_fn(*([tmb] + tmbln_list + tmbrn_list))[0]

                if np.isnan(cost) or np.isinf(cost):
                    print('bad cost detected! Cost is ' + str(cost))
                    return get_best_metric(history_valid_hits)

                if uidx % params[DISP_FREQ] == 0:
                    print('Epoch ', epoch, 'Iter', uidx, 'Cost ', cost)

                if uidx % params[VALID_FREQ] == 0:
                    print('Epoch ', epoch, 'Iter', uidx, 'Cost ', cost)

                    # print("Epoch {} of {} uidx {} took {:.3f}s".format(
                    #     epoch + 1, params[NUM_EPOCHS], uidx, time.time() - start_time))
                    if len(history_epoch_times) > 0:
                        print ("  Average epoch time - {:.3f}s".format(np.mean(history_epoch_times)))
                    print("  Time since start - {:.3f}s".format(time.time() - train_start_time))

                    print("  Train Minibatch Metrics")
                    train_hits10 = get_batch_metrics(tmb, sr_index, scores_fn, False)
                    print('')
                    print("  Validation data Metrics")
                    valid_hits10 = get_batch_metrics(valid_triples, sr_index, scores_fn, True)
                    print('')
                    print("  Test data Metrics")
                    test_hits10 = get_batch_metrics(test_triples, sr_index, scores_fn, True)

                    if (best_p is None) or (len(history_valid_hits) > 0 and valid_hits10 >= np.max(history_valid_hits)):
                        print("found best params yet")
                        best_p = utils.get_params(model)

                    history_valid_hits.append(valid_hits10)
                    history_test_hits.append(test_hits10)

                if uidx % params[SAVE_FREQ] == 0:
                    if best_p is None:
                        all_params = utils.get_params(model)
                    else:
                        all_params = best_p

                    utils.save(params[SAVETO_FILE], all_params)

                uidx += 1

            history_epoch_times.append(time.time() - epoch_start_time)

    except KeyboardInterrupt:
        print("training interrupted")

    return model, get_best_metric(history_valid_hits), train_fn, scores_fn
