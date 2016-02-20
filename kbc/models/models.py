from abc import ABCMeta, abstractmethod

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import lasagne

from kbc.utils import config

import time
import logging
from collections import OrderedDict


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
    return T.sum(out * (out > 0)), out > 0


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
# ----------------------------------------------------------------------------


# Embeddings class -----------------------------------------------------------
class Embeddings(object):
    """ Class for the embeddings matrix. """

    def __init__(self, rng, N, D, tag=''):
        """
        Constructor

        :param rng: numpy.random module for number generation.
        :param N: number of entities, relations or both.
        :param D: dimension of the embeddings
        :param tag: name of the embeddings.
        """
        self.N = N
        self.D = D
        wbound = np.sqrt(6. / D)
        W_values = rng.uniform(low=-wbound, high=wbound, size=(N, D))
        W_norm = np.sqrt(np.sum(W_values ** 2, axis=1)).reshape((N, 1))
        W_values = W_values / W_norm
        self.E = theano.shared(value=W_values, name='E' + tag)
        # Define a normalization function with respect to the L_2 norm of the
        # embedding vectors.
        self.updates = OrderedDict({self.E: self.E / T.sqrt(T.sum(self.E ** 2, axis=1)).reshape((N, 1))})
        self.normalize = theano.function([], [], updates=self.updates)
# ----------------------------------------------------------------------------


class Model(object):
    __metaclass__ = ABCMeta


class Model3(Model):

    def __init__(self, n_entities, n_relations, n_dim=10):
        self.rng = np.random
        self.srng = MRG_RandomStreams

        self.n_entities = n_entities
        self.n_relations = n_relations
        self.n_dim = n_dim

        self.r_embedding = Embeddings(self.rng, n_relations, n_dim, 'rel')
        self.e_embedding = Embeddings(self.rng, n_entities, n_dim, 'ent')
        self.embeddings = [self.e_embedding, self.r_embedding]

    def cost(self, pos_triples, neg_triples, marge=1.0):
        e_ss = pos_triples[:, 0]
        rs = pos_triples[:, 1]
        e_os = pos_triples[:, 2]

        e_ssn = neg_triples[:, 0]
        rsn = neg_triples[:, 1]
        e_osn = neg_triples[:, 2]

        e_embedding, r_embedding = self.embeddings

        pos = T.sum(e_embedding.E[e_ss] * r_embedding.E[rs] * e_embedding.E[e_os], axis=1)
        neg = T.sum(e_embedding.E[e_ssn] * r_embedding.E[rsn] * e_embedding.E[e_osn], axis=1)

        cost, out = margin_cost(pos, neg, marge)
        return cost, out

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

    def train_fn(self, lrate=0.01, marge=1.0):
        pos_triples = T.imatrix()
        neg_triples = T.imatrix()

        cost, out = self.cost(pos_triples, neg_triples, marge)
        params = [self.embeddings[0].E, self.embeddings[1].E]
        # updates = lasagne.updates.sgd(cost, params, lrate)
        updates = lasagne.updates.adagrad(cost, params, lrate) # way faster convergence
        # updates = lasagne.updates.sgd(cost, params, lrate) # slow like sgd (probably slower)

        return theano.function([pos_triples, neg_triples], [cost, out], updates=updates)


if __name__ == "__main__":
    # @TODO: write some unit tests
    pass
