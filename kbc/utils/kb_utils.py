import time
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


def train(model, train_triples, valid_triples, test_triples, sr_index, params):
    rng = np.random
    n_entities, n_relations = model.n_entities, model.n_relations

    train_fn = model.train_fn(num_neg=params[NUM_NEG], lrate=params[LEARNING_RATE], marge=params[MARGE])
    ranks_fn = model.ranks_fn()
    scores_fn = model.scores_fn()

    uidx = 1
    best_p = None
    history_valid = []
    history_test = []
    bins = [1, 11, 21, 31, 51, 101, 1001, 10001, 20000]
    print("Training on %d triples" % len(train_triples))
    print("The eval is being printed with number of items the bins -> %s" % bins)
    try:
        # We iterate over epochs:
        for epoch in range(params[NUM_EPOCHS]):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for _, train_index in utils.get_minibatches_idx(len(train_triples), params[BATCH_SIZE], False):
                # Normalize the entity embeddings
                if params[IS_NORMALIZED]:
                    model.normalize()

                tmb = train_triples[train_index]

                # generating negative examples replacing left entity
                tmbln_list = [rng.randint(0, n_entities, tmb.shape[0]).astype(dtype=tmb.dtype) for i in xrange(params[NUM_NEG])]

                # generating negative examples replacing right entity
                tmbrn_list = [rng.randint(0, n_entities, tmb.shape[0]).astype(dtype=tmb.dtype) for i in xrange(params[NUM_NEG])]

                cost_test = train_fn(*([tmb] + tmbln_list + tmbrn_list))[0]
                cost = cost_test

                # print('Epoch ', epoch, 'Iter', uidx, 'Cost ', cost)

                if np.isnan(cost) or np.isinf(cost):
                    print('bad cost detected! Cost is ' + str(cost))
                    return

                if uidx % params[DISP_FREQ] == 0:
                    print('Epoch ', epoch, 'Iter', uidx, 'Cost ', cost)

                if uidx % params[VALID_FREQ] == 0:
                    train_ranks = get_ranks(scores_fn(tmb)[0], tmb[:, 2])
                    valid_scores = scores_fn(valid_triples)[0]
                    test_scores = scores_fn(test_triples)[0]
                    valid_ranks = get_ranks(valid_scores, valid_triples[:, 2])
                    test_ranks = get_ranks(test_scores, test_triples[:, 2])
                    valid_disc_ranks = get_discounted_ranks(valid_scores, valid_triples, sr_index)
                    test_disc_ranks = get_discounted_ranks(test_scores, test_triples, sr_index)

                    train_err = np.mean(train_ranks)
                    valid_err = np.mean(valid_ranks)
                    test_err = np.mean(test_ranks)
                    valid_disc_err = np.mean(valid_disc_ranks)
                    test_disc_err = np.mean(test_disc_ranks)

                    train_hits10 = float((train_ranks <= 10).astype('float32').sum()) / train_ranks.shape[0]
                    valid_hits10 = float((valid_ranks <= 10).astype('float32').sum()) / valid_ranks.shape[0]
                    test_hits10 = float((test_ranks <= 10).astype('float32').sum()) / test_ranks.shape[0]
                    valid_disc_hits10 = float((valid_disc_ranks <= 10).astype('float32').sum()) / valid_ranks.shape[0]
                    test_disc_hits10 = float((test_disc_ranks <= 10).astype('float32').sum()) / test_ranks.shape[0]

                    train_dist = np.histogram(train_ranks, bins)
                    valid_dist = np.histogram(valid_ranks, bins)
                    test_dist = np.histogram(test_ranks, bins)

                    # Then we print the results for this epoch:
                    print("Epoch {} of {} uidx {} took {:.3f}s".format(
                        epoch + 1, params[NUM_EPOCHS], uidx, time.time() - start_time))
                    print("  mean training triples rank: %f" % train_err)
                    print("  mean validation triples rank: %f" % valid_err)
                    print("  mean validation triples discounted rank: %f" % valid_disc_err)
                    print("  mean test triples discounted rank: %f" % test_disc_err)
                    print("  mean test triples rank: %f" % test_err)
                    print("  training triples rank dist: %s" % train_dist[0])
                    print("  validation triples rank dist: %s" % valid_dist[0])
                    print("  test triples rank dist: %s" % test_dist[0])
                    print("  training triples hits@10: %s" % train_hits10)
                    print("  validation triples hits@10: %s" % valid_hits10)
                    print("  test triples hits@10: %s" % test_hits10)
                    print("  validation triples discounted hits@10: %s" % valid_disc_hits10)
                    print("  test triples discounted hits@10: %s" % test_disc_hits10)

                    if (best_p is None) or (len(history_valid) > 0 and valid_hits10 >= np.max(history_valid)):
                        print("found best params yet")
                        best_p = utils.get_params(model)

                    history_valid.append(valid_hits10)
                    history_test.append(test_hits10)

                if uidx % params[SAVE_FREQ] == 0:
                    if best_p is None:
                        all_params = utils.get_params(model)
                    else:
                        all_params = best_p

                    # utils.save(params[SAVETO_FILE], all_params, params)
                    utils.save(params[SAVETO_FILE], all_params)

                uidx += 1

    except KeyboardInterrupt:
        print("training interrupted")

    valid_ranks = ranks_fn(valid_triples)[0]
    test_ranks = ranks_fn(test_triples)[0]

    train_err = np.mean(train_ranks)
    valid_err = np.mean(valid_ranks)
    test_err = np.mean(test_ranks)

    valid_hits10 = float((valid_ranks <= 10).astype('float32').sum()) / valid_ranks.shape[0]
    test_hits10 = float((test_ranks <= 10).astype('float32').sum()) / test_ranks.shape[0]

    print("\nresults after training")
    # print("  mean training triples rank: %f" % train_err)
    print("  mean validation triples rank: %f" % valid_err)
    print("  mean test triples rank: %f" % test_err)

    print("  validation triples hits@10: %s" % valid_hits10)
    print("  test triples hits@10: %s" % test_hits10)

    # return all stuff needed for debugging
    return model, train_ranks, valid_ranks

