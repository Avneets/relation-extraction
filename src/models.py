from abc import ABCMeta, abstractmethod
import theano
import theano.tensor as T
import numpy as np


class Model(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def score(self, sub, obj, rel): pass

    @abstractmethod
    def get_L1(self): pass

    @abstractmethod
    def get_L2(self): pass


class DistMult(Model):

    def __init__(self, n_entities, n_relations, n_dim=10, entity_embeddings=None, relation_embeddings=None):
        """
        The model for DistMult which scores a triple (s, r, o) by v_r . (v_s * v_o)
        where v_r, v_s, v_o are embeddings for relation, subject and object respectively and '.' and '*'
        denote inner product and element-wise product respectively.

        :type n_entities: int
        :param n_entities: size of the entity vocabulary

         :type n_relations: int
         :param n_relations: size of the relation vocabulary

        :type n_dim: int
        :param n_dim: size of embeddings

        :type entity_embeddings: theano.tensor.matrix
        :param entity_embeddings: entity embedding matrix (symbolic)

        :type relation_embeddings: theano.tensor.matrix
        :param relation_embeddings: relation embedding matrix (symbolic)

        :return:None
        """

        rng = np.random

        if entity_embeddings is None:
            # Entity embedding initialisation. Change later!
            embeddings_init = np.asarray(rng.randn(n_entities, n_dim), dtype=theano.config.floatX)
            entity_embeddings = theano.shared(embeddings_init, name='entity_embeddings', borrow=True)

        if relation_embeddings is None:
            # Relation embedding initialisation. Change later!
            embeddings_init = np.asarray(rng.randn(n_relations, n_dim), dtype=theano.config.floatX)
            relation_embeddings = theano.shared(embeddings_init, name='entity_embeddings', borrow=True)

        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings

        self.params = [self.entity_embeddings, self.relation_embeddings]

        self.L2 = (self.entity_embeddings ** 2).sum() + (self.relation_embeddings ** 2).sum()
        self.L1 = abs(self.entity_embeddings).sum() + abs(self.relation_embeddings).sum()

    def score(self, sub, obj, rel):
        """
        Returns the score of the triple (subject, relation, object) under the DistMult model.

        :type sub: theano.tensor.TensorType
        :param sub: subject entity index (symbolic variable)

        :type obj: theano.tensor.TensorType
        :param obj: object entity index (symbolic variable)

        :type rel: theano.tensor.TensorType
        :param rel: relation index (symbolic variable)
        """

        return T.dot(self.relation_embeddings[rel],
                     (self.entity_embeddings[sub] * self.entity_embeddings[obj]))

    def get_params(self): return self.params

    def get_L1(self): return self.L1

    def get_L2(self): return self.L2


def conditional_score(model, sub, obj, rel, neg_samples, issub):
    """
    Gives the conditional score (not probability) of object given subject and relation

    :type model: Model
    :param model: the model to be used

    :type sub: theano.tensor.TensorType
    :param sub: subject entity index (symbolic variable)

    :type obj: theano.tensor.TensorType
    :param obj: object entity index (symbolic variable)

    :type rel: theano.tensor.TensorType
    :param rel: relation index (symbolic variable)

    :type neg_samples: theano.tensor.TensorType
    :param neg_samples: a symbolic array denoting the indices of objects which can
    be used in the loglikelihood scoring function

    :type issub: bool
    :param issub: if True, returns conditional score for subject, else for object
    """

    assert isinstance(model, Model)

    numer = T.exp(model.score(sub, obj, rel))
    outputs_info=T.as_tensor_variable(np.array(0, dtype=numer.dtype))

    if issub:
        denom = theano.scan(fn=lambda sample_index, sum_upto: sum_upto + T.exp(model.score(sample_index, obj, rel)),
                            outputs_info=outputs_info,
                            sequences=neg_samples)
    else:
        denom = theano.scan(fn=lambda sample_index, sum_upto: sum_upto + T.exp(model.score(sub, sample_index, rel)),
                            outputs_info=outputs_info,
                            sequences=neg_samples)
    return numer / denom