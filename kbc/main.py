import time
from collections import OrderedDict
import six.moves.cPickle as pickle

import numpy as np

from kbc.reader import KBReader
from kbc.utils import config
from kbc.models import models


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
    embeddings = model.embeddings
    params = OrderedDict()
    for embedding in embeddings:
        params[embedding.tag] = embedding.E.get_value()
    return params


def save(saveto, params, options):
    print('Saving params...')

    np.savez(saveto, **params)
    pickle.dump(options, open('%s.pkl' % saveto, 'wb'), -1)
    print('Done')


def load_params(path):
    pp = np.load(path)
    return pp


def main(model_file='model.npz',
         saveto='model.npz',
         reload_model=False,
         num_epochs=500,
         full_train=False,
         num_train=5000,
         dim_emb=50,
         L1_reg=0.0,
         L2_reg=0.0,
         is_normalized=True,
         learning_rate=0.01,
         batch_size=128,
         valid_freq=1000,
         disp_freq=200,
         save_freq=5000,
         valid_frac=0.2,
         test_frac=0.2,
         params=None):
    # Load the dataset
    print("Loading data...")
    model_options = locals().copy()  # has all the parameters required for the model
    rng = np.random
    print("model options", model_options)
    debug_data = {}

    print("loading train KB data")
    train_dataset = KBReader.read_data(config.KBTrainFile)
    train_dataset.print_set_statistics()
    print("loading validation KB data")
    valid_dataset = KBReader.read_data(
        filepath=config.KBValidationFile,
        entity_dict=train_dataset.entity_index,
        entity_pair_dict=train_dataset.entity_pair_index,
        relation_dict=train_dataset.relation_index,
        add_new=False
    )
    valid_dataset.print_set_statistics()
    print("loading test KB data")
    test_dataset = KBReader.read_data(
        filepath=config.KBTestFile,
        entity_dict=train_dataset.entity_index,
        entity_pair_dict=train_dataset.entity_pair_index,
        relation_dict=train_dataset.relation_index,
        add_new=False
    )
    test_dataset.print_set_statistics()
    print("generating full train data")
    if full_train:
        train_triples = train_dataset.generate_batch(batch_size=-1)[0].astype(np.int32)
    else:
        train_triples = train_dataset.generate_batch(batch_size=num_train)[0].astype(np.int32)
    valid_triples = valid_dataset.generate_batch(batch_size=int(valid_frac * valid_dataset.n_triples))[0].astype(np.int32)
    # valid_triples = train_dataset.generate_batch(batch_size=num_train)[0].astype(np.int32)
    test_triples = test_dataset.generate_batch(batch_size=int(test_frac * test_dataset.n_triples))[0].astype(np.int32)

    n_entities = train_dataset.n_entities
    n_relations = train_dataset.n_relations

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")

    params_init = None
    if reload_model:
        params_init = load_params(model_file)
    model = models.Model2(n_entities, n_relations, dim_emb, params_init, is_normalized=is_normalized, L1_reg=L1_reg, L2_reg=L2_reg)
    # model = models.Model2plus3(n_entities, n_relations, dim_emb, params_init, is_normalized=is_normalized, L1_reg=L1_reg, L2_reg=L2_reg)
    # model = models.Model3(n_entities, n_relations, dim_emb, params_init, is_normalized=is_normalized, L1_reg=L1_reg, L2_reg=L2_reg)
    train_fn = model.train_fn(learning_rate, marge=1.0)
    ranks_fn = model.ranks_fn()

    uidx = 0
    best_p = None
    history_valid = []
    history_test = []
    bins = [1, 11, 101, 1001, 10001, 20000]
    print("Training on %d triples" % len(train_triples))
    print("The eval is being printed with number of items the bins -> %s" % bins)
    try:
        # We iterate over epochs:
        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for _, train_index in get_minibatches_idx(len(train_triples), batch_size, False):
                # Normalize the entity embeddings
                if is_normalized:
                    model.normalize()

                tmb = train_triples[train_index]

                # generating negative examples replacing left entity
                tmbln = np.empty(tmb.shape, dtype=tmb.dtype)
                tmbln[:, [1, 2]] = tmb[:, [1, 2]]
                tmbln[:, 0] = rng.randint(0, n_entities, tmb.shape[0])

                tmbrn = np.empty(tmb.shape, dtype=tmb.dtype)
                tmbrn[:, [0, 1]] = tmb[:, [0, 1]]
                tmbrn[:, 2] = rng.randint(0, n_entities, tmb.shape[0])

                costl = train_fn(tmb, tmbln)[0]
                costr = train_fn(tmb, tmbrn)[0]

                cost = costl + costr

                # print('Epoch ', epoch, 'Iter', uidx, 'Cost ', cost)

                if np.isnan(cost) or np.isinf(cost):
                    print('bad cost detected! Cost is ' + str(cost))
                    return

                if uidx % disp_freq == 0:
                    print('Epoch ', epoch, 'Iter', uidx, 'Cost ', cost)

                if uidx % valid_freq == 0:
                    train_ranks = ranks_fn(tmb)[0]
                    valid_ranks = ranks_fn(valid_triples)[0]
                    test_ranks = ranks_fn(test_triples)[0]

                    train_err = np.mean(train_ranks)
                    valid_err = np.mean(valid_ranks)
                    test_err = np.mean(test_ranks)

                    train_hits10 = float((train_ranks <= 10).astype('float32').sum()) / train_ranks.shape[0]
                    valid_hits10 = float((valid_ranks <= 10).astype('float32').sum()) / valid_ranks.shape[0]
                    test_hits10 = float((test_ranks <= 10).astype('float32').sum()) / test_ranks.shape[0]

                    train_dist = np.histogram(train_ranks, bins)
                    valid_dist = np.histogram(valid_ranks, bins)
                    test_dist = np.histogram(test_ranks, bins)

                    # Then we print the results for this epoch:
                    print("Epoch {} of {} uidx {} took {:.3f}s".format(
                        epoch + 1, num_epochs, uidx, time.time() - start_time))
                    print("  mean training triples rank: %f" % train_err)
                    print("  mean validation triples rank: %f" % valid_err)
                    print("  mean test triples rank: %f" % test_err)
                    print("  training triples rank dist: %s" % train_dist[0])
                    print("  validation triples rank dist: %s" % valid_dist[0])
                    print("  test triples rank dist: %s" % test_dist[0])
                    print("  training triples hits@10: %s" % train_hits10)
                    print("  validation triples hits@10: %s" % valid_hits10)
                    print("  test triples hits@10: %s" % test_hits10)

                    if (best_p is None) or (len(history_valid) > 0 and valid_hits10 >= np.max(history_valid)):
                        print("found best params yet")
                        best_p = get_params(model)

                    history_valid.append(valid_hits10)
                    history_test.append(test_hits10)

                if uidx % save_freq == 0:
                    if best_p is None:
                        params = get_params(model)
                    else:
                        params = best_p

                    save(saveto, params, model_options)



                uidx += 1
    except KeyboardInterrupt:
        print("training interrupted")

    # After training, we compute and print the test error:
    # train_triples are pretty big. So split and calculate ranks

    # train_ranks = []
    # print("Evaluating on all training data")
    # for _, train_index in get_minibatches_idx(len(train_triples), batch_size, False):
    #     train_ranks += list(ranks_fn(train_triples[train_index])[0])
    valid_ranks = ranks_fn(valid_triples)
    test_ranks = ranks_fn(test_triples)

    train_err = np.mean(train_ranks)
    valid_err = np.mean(valid_ranks)
    test_err = np.mean(test_ranks)

    print("\nresults after training")
    # print("  mean training triples rank: %f" % train_err)
    print("  mean validation triples rank: %f" % valid_err)
    print("  mean test triples rank: %f" % test_err)

    # return all stuff needed for debugging
    return model, train_ranks, valid_ranks


if __name__ == '__main__':
    rng = np.random
    # server model
    model, train_ranks, valid_ranks = main(model_file='model2_150epochs_noreg.npz',
                 saveto='model2_150epochs_noreg',
                 reload_model=False,
                 num_epochs=150,
                 full_train=True,
                 valid_freq=10000,
                 disp_freq=1000,
                 save_freq=20000,
                 valid_frac=1.0,
                 test_frac=1.0,
                 L1_reg=0.0,
                 L2_reg=0.0,
                 is_normalized=True
                 )
    # local model
    # ranks = main(model_file='model2+3test.npz',
    #              saveto='model2+3test',
    #              reload_model=False,
    #              num_epochs=10,
    #              num_train=5000,
    #              valid_freq=200,
    #              disp_freq=20,
    #              save_freq=1000,
    #              valid_frac=1.0,
    #              test_frac=1.0)
