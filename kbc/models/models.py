from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import lasagne


# Similarity functions -------------------------------------------------------
def L1sim(left, right):
    return - T.sum(T.abs_(left - right), axis=1)


def L2sim(left, right):
    return - T.sqrt(T.sum(T.sqr(left - right), axis=1))


def Dotsim(left, right):
    return T.sum(left * right, axis=1)


def DistModSim(l, o, r):
    return T.sum(l * o * r, axis=1)
# -----------------------------------------------------------------------------


# Cost Functions -----------------------------------------------------------
def margin_cost(pos, neg, marge=1.0):
    """

    :param pos: positive instance tensor array
    :param neg: corresponding negative instance tensor array
    :param marge: margin (typically 1.0)
    :return: sum_i max(neg[i] - pos[i] + marge, 0) - margin based error function
    """
    out = neg - pos + marge
    return T.sum(out * (out > 0))


def get_L1(W):
    return T.sum(T.abs_(W))


def get_L2(W):
    return T.sum(W ** 2)


def parse_embeddings(embeddings):
    if type(embeddings) == list:
        if len(embeddings) == 2:
            ent_embedding = embeddings[0]
            rel_embedding = embeddings[1]
            return ent_embedding, rel_embedding
        elif len(embeddings) == 3:
            ent_embedding = embeddings[0]
            rell_embedding = embeddings[1]
            relr_embedding = embeddings[2]
            return ent_embedding, rell_embedding, relr_embedding
    else:
        print("couldn't read the embeddings")
        exit()


def batch_marge_cost(score_batch, pos_triples, neg_entities_list, is_subject=True, marge=1.0):
    e_ss = pos_triples[:, 0]
    rs = pos_triples[:, 1]
    e_os = pos_triples[:, 2]

    cost = 0

    if is_subject:
        for e_ssn in neg_entities_list:
            pos = score_batch(e_ss, rs, e_os)
            neg = score_batch(e_ssn, rs, e_os)
            cost += margin_cost(pos, neg, marge)
    else:
        for e_osn in neg_entities_list:
            pos = score_batch(e_ss, rs, e_os)
            neg = score_batch(e_ss, rs, e_osn)
            cost += margin_cost(pos, neg, marge)

    return cost


def train_fn(cost_fns, params, num_neg=10, lrate=0.01, marge=1.0):
        pos_triples = T.imatrix()
        sub_neg_entities_list = [T.ivector() for i in xrange(num_neg)]
        obj_neg_entities_list = [T.ivector() for i in xrange(num_neg)]

        cost_sub = 0
        cost_obj = 0
        for cost_fn in cost_fns:
            cost_sub += cost_fn(pos_triples, sub_neg_entities_list, True, marge)
            cost_obj += cost_fn(pos_triples, obj_neg_entities_list, False, marge)
        cost = cost_sub + cost_obj
        updates = lasagne.updates.adagrad(cost, params, lrate)

        return theano.function([pos_triples] + sub_neg_entities_list + obj_neg_entities_list, [cost], updates=updates)
# ----------------------------------------------------------------------------


# Model utility functions -----------------------------------------------------------

def get_embeddings_dict(model):
    embeddings_dict = OrderedDict()
    for embedding in model.embeddings:
        embeddings_dict[embedding.tag] = embedding.E
    return embeddings_dict

# --------------------------------------------------------------------------------------


# Embeddings class -----------------------------------------------------------
class Embeddings(object):
    """ Class for the embeddings matrix. """

    def __init__(self, rng, N, D, tag='', W_init=None, init_normalize=True):
        """
        Constructor

        :param rng: numpy.random module for number generation.
        :param N: number of entities, relations or both.
        :param D: dimension of the embeddings
        :param tag: name of the embeddings.
        """
        self.N = N
        self.D = D
        self.tag = tag
        if W_init is None:
            wbound = np.sqrt(6. / D)
            W_values = rng.uniform(low=-wbound, high=wbound, size=(N, D))
            if init_normalize:
                W_norm = np.sqrt(np.sum(W_values ** 2, axis=1)).reshape((N, 1))
                W_values = W_values / W_norm
        else:
            W_values = W_init
        self.E = theano.shared(value=W_values, name=tag)
        # Define a normalization function with respect to the L_2 norm of the
        # embedding vectors.
        self.updates = OrderedDict({self.E: self.E / T.sqrt(T.sum(self.E ** 2, axis=1)).reshape((N, 1))})
        self.normalize = theano.function([], [], updates=self.updates)
# ----------------------------------------------------------------------------


class Model(object):
    __metaclass__ = ABCMeta


class Model3(Model):

    def __init__(self, n_entities, n_relations, n_dim=10, params=None, is_normalized=True, L1_reg=0.0, L2_reg=0.0):
        self.rng = np.random
        self.srng = MRG_RandomStreams

        self.n_entities = n_entities
        self.n_relations = n_relations
        self.n_dim = n_dim
        self.is_normalized = is_normalized
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg

        self.e_embedding_tag = 'model3-ent'
        self.r_embedding_tag = 'model3-rel'

        e_embedding_init=None
        r_embedding_init=None
        if params is not None:
            e_embedding_init = params[self.e_embedding_tag]
            r_embedding_init = params[self.r_embedding_tag]
        self.e_embedding = Embeddings(self.rng, n_entities, n_dim, self.e_embedding_tag, e_embedding_init, is_normalized)
        self.r_embedding = Embeddings(self.rng, n_relations, n_dim, self.r_embedding_tag, r_embedding_init, is_normalized)
        self.embeddings = [self.e_embedding, self.r_embedding]
        self.all_params_dict = get_embeddings_dict(self)

    def normalize(self):
        self.e_embedding.normalize()

    def score_batch(self, e_ss, rs, e_os):
        e_embedding, r_embedding = self.embeddings
        return T.sum(e_embedding.E[e_ss] * r_embedding.E[rs] * e_embedding.E[e_os], axis=1)

    def cost(self, pos_triples, neg_entities_list, is_subject=True, marge=1.0):
        cost = batch_marge_cost(self.score_batch, pos_triples, neg_entities_list, is_subject, marge)

        if self.L1_reg > 0.:
            for embedding in self.embeddings:
                cost += self.L1_reg * get_L1(embedding.E)
        if self.L2_reg > 0.:
            for embedding in self.embeddings:
                cost += self.L2_reg * get_L2(embedding.E)

        return cost

    def train_fn(self, num_neg=10, lrate=0.01, marge=1.0):
        params = [self.embeddings[0].E, self.embeddings[1].E]
        return train_fn([self.cost], params, num_neg, lrate, marge)

    def all_ent_scores(self, in_triples):
        e_ss = in_triples[:, 0]
        rs = in_triples[:, 1]
        e_embedding, r_embedding = self.embeddings
        scores = T.dot( (e_embedding.E[e_ss]*r_embedding.E[rs]), e_embedding.E.T )
        return scores

    def scores_fn(self):
        in_triples = T.imatrix()
        return theano.function([in_triples], [self.all_ent_scores(in_triples)])

    def ranks_fn(self):
        in_triples = T.imatrix()
        e_ss = in_triples[:, 0]
        rs = in_triples[:, 1]
        e_os = in_triples[:, 2]

        e_embedding, r_embedding = self.embeddings

        scores = T.dot( (e_embedding.E[e_ss]*r_embedding.E[rs]), e_embedding.E.T )
        e_os_scores = scores[T.arange(scores.shape[0]), e_os]
        ranks_os = ( scores >= e_os_scores.reshape((-1, 1)) ).sum(axis=1)

        return theano.function([in_triples], [ranks_os])


class Model2(Model):

    def __init__(self, n_entities, n_relations, n_dim=10, params=None, is_normalized=True, L1_reg=0.0, L2_reg=0.0):
        self.rng = np.random
        self.srng = MRG_RandomStreams

        self.n_entities = n_entities
        self.n_relations = n_relations
        self.n_dim = n_dim
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg

        self.e_embedding_tag = 'model2-ent'
        self.rl_embedding_tag = 'model2-rell'
        self.rr_embedding_tag = 'model2-relr'

        e_embedding_init=None
        rl_embedding_init=None
        rr_embedding_init=None
        if params is not None:
            e_embedding_init = params[self.e_embedding_tag]
            rl_embedding_init = params[self.rl_embedding_tag]
            rr_embedding_init = params[self.rr_embedding_tag]

        self.rl_embedding = Embeddings(self.rng, n_relations, n_dim, self.rl_embedding_tag, rl_embedding_init, is_normalized)
        self.rr_embedding = Embeddings(self.rng, n_relations, n_dim, self.rr_embedding_tag, rr_embedding_init, is_normalized)
        self.e_embedding = Embeddings(self.rng, n_entities, n_dim, self.e_embedding_tag, e_embedding_init, is_normalized)
        self.embeddings = [self.e_embedding, self.rl_embedding, self.rr_embedding]
        self.all_params_dict = get_embeddings_dict(self)

    def normalize(self):
        self.e_embedding.normalize()

    def score_batch(self, e_ss, rs, e_os):
        e_embedding, rl_embedding, rr_embedding = self.embeddings
        return T.sum(e_embedding.E[e_ss] * rl_embedding.E[rs], axis=1) + T.sum(rr_embedding.E[rs] * e_embedding.E[e_os], axis=1)

    def cost(self, pos_triples, neg_entities_list, is_subject=True, marge=1.0):
        cost = batch_marge_cost(self.score_batch, pos_triples, neg_entities_list, is_subject, marge)
        if self.L1_reg > 0.:
            for embedding in self.embeddings:
                cost += self.L1_reg * get_L1(embedding.E)
        if self.L2_reg > 0.:
            for embedding in self.embeddings:
                cost += self.L2_reg * get_L2(embedding.E)
        return cost

    def train_fn(self, num_neg=10, lrate=0.01, marge=1.0):
        params = [self.embeddings[0].E, self.embeddings[1].E, self.embeddings[2].E]
        return train_fn([self.cost], params, num_neg, lrate, marge)

    def all_ent_scores(self, in_triples):
        rs = in_triples[:, 1]
        e_embedding, _, rr_embedding = self.embeddings
        scores = T.dot( (rr_embedding.E[rs]), e_embedding.E.T )
        return scores

    def scores_fn(self):
        in_triples = T.imatrix()
        return theano.function([in_triples], [self.all_ent_scores(in_triples)])

    def ranks_fn(self):
        in_triples = T.imatrix()
        e_ss = in_triples[:, 0]
        rs = in_triples[:, 1]
        e_os = in_triples[:, 2]

        e_embedding, rl_embedding, rr_embedding = self.embeddings

        scores = T.dot( (rr_embedding.E[rs]), e_embedding.E.T )
        e_os_scores = scores[T.arange(scores.shape[0]), e_os]
        ranks_os = ( scores >= e_os_scores.reshape((-1, 1)) ).sum(axis=1)

        return theano.function([in_triples], [ranks_os])


class Model2plus3(Model):

    def __init__(self, n_entities, n_relations, n_dim=10, params=None, is_normalized=True, L1_reg=0.0, L2_reg=0.0):
        self.rng = np.random
        self.srng = MRG_RandomStreams

        self.n_entities = n_entities
        self.n_relations = n_relations
        self.n_dim = n_dim
        self.L1_reg = L1_reg
        self.L2_reg = L2_reg

        self.model2 = Model2(n_entities, n_relations, n_dim, params, is_normalized, L1_reg, L2_reg)
        self.model3 = Model3(n_entities, n_relations, n_dim, params, is_normalized, L1_reg, L2_reg)

        self.embeddings = self.model2.embeddings + self.model3.embeddings
        self.all_params_dict = get_embeddings_dict(self)

    def normalize(self):
        self.model2.normalize()
        self.model3.normalize()

    def scores_fn(self):
        in_triples = T.imatrix()
        scores = self.model2.all_ent_scores(in_triples) + self.model3.all_ent_scores(in_triples)
        return theano.function([in_triples], [scores])

    def ranks_fn(self):
        in_triples = T.imatrix()
        e_ss = in_triples[:, 0]
        rs = in_triples[:, 1]
        e_os = in_triples[:, 2]

        scores2 = self.model2.all_ent_scores(in_triples)
        scores3 = self.model3.all_ent_scores(in_triples)
        scores = scores2 + scores3
        e_os_scores = scores[T.arange(scores.shape[0]), e_os]
        ranks_os = ( scores >= e_os_scores.reshape((-1, 1)) ).sum(axis=1)

        return theano.function([in_triples], [ranks_os])

    def train_fn(self, num_neg=10, lrate=0.01, marge=1.0):
        params = [embedding.E for embedding in self.embeddings]
        return train_fn([self.model2.cost, self.model3.cost], params, num_neg, lrate, marge)


if __name__ == "__main__":
    # @TODO: write some unit tests
    model3 = Model3(14000, 300, 40)
