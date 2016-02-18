from abc import ABCMeta, abstractmethod
import theano
import theano.tensor as T
import numpy as np
import time
import config
import logging


class Model(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.L1_reg = 0.0
        self.L2_reg = 0.0

    @abstractmethod
    def score(self, sub, obj, rel): pass

    @abstractmethod
    def get_L1(self): pass

    @abstractmethod
    def get_L2(self): pass

    def log_conditional_score(self, sub, rel, obj, neg_samples, issub):
        """
        Gives the conditional score (not probability) of object(subject) given subject(object) and relation

        :type model: Model
        :param model: the model to be used

        :type sub: theano.tensor.TensorType
        :param sub: subject entity index (symbolic variable)

        :type rel: theano.tensor.TensorType
        :param rel: relation index (symbolic variable)

        :type obj: theano.tensor.TensorType
        :param obj: object entity index (symbolic variable)

        :type neg_samples: theano.tensor.TensorType
        :param neg_samples: a symbolic array denoting the indices of objects which can
        be used in the loglikelihood scoring function

        :type issub: bool
        :param issub: if True, returns conditional score for subject, else for object
        """

        assert isinstance(self, Model)

        numer = T.exp(self.score(sub, rel, obj))
        outputs_info=T.as_tensor_variable(np.array(0, dtype=theano.config.floatX))

        if issub:
            results, updates = theano.scan(fn=lambda sample_index, sum_upto: sum_upto + T.exp(self.score(sample_index, rel, obj)),
                                outputs_info=outputs_info,
                                sequences=neg_samples)
            denom = results[-1]
        else:
            results, updates = theano.scan(fn=lambda sample_index, sum_upto: sum_upto + T.exp(self.score(sub, rel, sample_index)),
                                outputs_info=outputs_info,
                                sequences=neg_samples)
            denom = results[-1]
        return T.log(numer / denom)

    def loss_on_set(self, in_triples, in_sub_neg_samples_list, in_obj_neg_samples_list):

        acc_loss_init = T.as_tensor_variable(np.array(0, dtype=theano.config.floatX))
        results, updates = theano.scan(
            fn=lambda in_triple, in_sub_neg_samples, in_obj_neg_samples, acc_loss:
            acc_loss + self.log_conditional_score(in_triple[0], in_triple[1], in_triple[2], in_sub_neg_samples, True) +
            self.log_conditional_score(in_triple[0], in_triple[1], in_triple[2], in_obj_neg_samples, False),
            outputs_info=acc_loss_init,
            sequences=[in_triples, in_sub_neg_samples_list, in_obj_neg_samples_list]
        )
        return -results[-1]

    def total_cost(self, in_triples, in_sub_neg_samples_list, in_obj_neg_samples_list):

        return (self.loss_on_set(in_triples, in_sub_neg_samples_list, in_obj_neg_samples_list) +
        self.L1_reg * self.get_L1() + self.L2_reg * self.get_L2())



class DistMult(Model):

    def __init__(self, n_entities, n_relations, n_dim=10, L1_reg=0.0, L2_reg=1.0, params=None):
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

        self.L1_reg = L1_reg
        self.L2_reg = L2_reg

        if (params is None) or ((config.ENTITY_EMBEDDINGS in params) and (params[config.ENTITY_EMBEDDINGS] is None)):
            # Entity embedding initialisation. Change later!
            ent_embeddings_init = np.asarray(rng.randn(n_entities, n_dim), dtype=theano.config.floatX)
        else:
            ent_embeddings_init = params[config.ENTITY_EMBEDDINGS]
        logging.debug("Initial mean and std of entity_embeddings = (%f, %f)" % (np.mean(ent_embeddings_init), np.std(ent_embeddings_init)))

        if (params is None) or ((config.RELATION_EMBEDDINGS in params) and (params[config.RELATION_EMBEDDINGS] is None)):
            # Relation embedding initialisation. Change later!
            rel_embeddings_init = np.asarray(rng.randn(n_relations, n_dim), dtype=theano.config.floatX)
        else:
            rel_embeddings_init = params[config.RELATION_EMBEDDINGS]
        logging.debug("Initial mean and std of relation_embeddings = (%f, %f)" % (np.mean(rel_embeddings_init), np.std(rel_embeddings_init)))

        entity_embeddings = theano.shared(ent_embeddings_init, name='entity_embeddings', borrow=True)
        relation_embeddings = theano.shared(rel_embeddings_init, name='relation_embeddings', borrow=True)
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings

        self.tparams = {config.ENTITY_EMBEDDINGS: self.entity_embeddings, config.RELATION_EMBEDDINGS: self.relation_embeddings}

        self.L2 = (self.entity_embeddings ** 2).sum() + (self.relation_embeddings ** 2).sum()
        self.L1 = abs(self.entity_embeddings).sum() + abs(self.relation_embeddings).sum()

    def get_tparams(self): return self.tparams

    def get_L1(self): return self.L1

    def get_L2(self): return self.L2

    def score(self, sub, rel, obj):
        """
        Returns the score of the triple (subject, relation, object) under the DistMult model.

        :type sub: theano.tensor.TensorType
        :param sub: subject entity index (symbolic variable)

        :type rel: theano.tensor.TensorType
        :param rel: relation index (symbolic variable)

        :type obj: theano.tensor.TensorType
        :param obj: object entity index (symbolic variable)
        """

        return T.dot(self.relation_embeddings[rel],
                     (self.entity_embeddings[sub] * self.entity_embeddings[obj]))

    def loss_on_set2(self, in_triples, sub_negs, obj_negs):
        e_ss = in_triples[:, 0]
        rs = in_triples[:, 1]
        e_os = in_triples[:, 2]

        sub_negs_size = sub_negs.size
        obj_negs_size = obj_negs.size

        test = self.entity_embeddings[sub_negs.reshape((1, sub_negs_size))]

        net_numer = (self.entity_embeddings[e_ss] *
         self.entity_embeddings[e_os] *
         self.relation_embeddings[rs]
         ).sum()
        # sum along row for dot product and along column for all triples
        net_denom_obj = T.log(
            T.exp(
                (self.entity_embeddings[e_os].repeat(sub_negs.shape[1], axis=0) *
                 self.entity_embeddings[sub_negs.reshape((sub_negs_size, ))] *
                 self.relation_embeddings[rs].repeat(sub_negs.shape[1], axis=0)
                 ).sum(axis=1).reshape((-1, sub_negs.shape[1]))
            ).sum(axis=1)
        ).sum() # log of product of denominators

        net_denom_sub = T.log(
            T.exp(
                (self.entity_embeddings[e_ss].repeat(obj_negs.shape[1], axis=0) *
                 self.entity_embeddings[obj_negs.reshape((obj_negs_size, ))] *
                 self.relation_embeddings[rs].repeat(obj_negs.shape[1], axis=0)
                 ).sum(axis=1).reshape((-1, obj_negs.shape[1]))
            ).sum(axis=1)
        ).sum() # log of product of denominators

        return -2 * net_numer + net_denom_sub + net_denom_obj

    def total_cost(self, in_triples, in_sub_neg_samples_list, in_obj_neg_samples_list):
        # loss_on_set2 is almost 10 times faster than loss_on_set. Override loss_on_set with loss_on_set2 later
        return (self.loss_on_set2(in_triples, in_sub_neg_samples_list, in_obj_neg_samples_list) +
        self.L1_reg * self.get_L1() + self.L2_reg * self.get_L2())

    def pred_ranks(self, in_triples):
        e_ss = in_triples[:, 0]
        rs = in_triples[:, 1]
        e_os = in_triples[:, 2]

        scores = T.dot( (self.entity_embeddings[e_ss]*self.relation_embeddings[rs]), self.entity_embeddings.T )
        e_os_scores = scores[T.arange(scores.shape[0]), e_os]
        ranks_os = T.cast( scores >= e_os_scores.reshape((-1,1)), dtype=theano.config.floatX ).sum(axis=1)

        return ranks_os

    def metrics(self, in_triples, k, n_entities, num_triples):
        # n_entities, num_triples - hack to over come the fact that T.cast(arr, recs)
        # need constant recs! Change this later.
        e_ss = in_triples[:, 0]
        rs = in_triples[:, 1]
        e_os = in_triples[:, 2]

        scores = T.dot( (self.entity_embeddings[e_ss]*self.relation_embeddings[rs]), self.entity_embeddings.T )
        e_os_scores = scores[T.arange(scores.shape[0]), e_os]
        ranks_os = T.cast( scores >= e_os_scores.reshape((-1,1)), dtype=theano.config.floatX ).sum(axis=1)

        mrr = T.mean(1 / ranks_os)
        hits = T.cast(T.nonzero(ranks_os <= k)[0].shape[0], dtype=theano.config.floatX) / T.cast(ranks_os.shape[0], dtype=theano.config.floatX)

        return {'mrr': mrr,
                'hits@k': hits
                }


if __name__ == "__main__":
    # A dummy model

    dist_mult_model = DistMult(30000, 1000)

    in_triples = T.imatrix()
    in_sub_neg_samples_mat = T.imatrix()
    in_obj_neg_samples_mat = T.imatrix()

    rng = np.random
    in_triples_init = np.concatenate(
        (rng.randint(0, 30000, (1000, 1)),
         rng.randint(0, 1000, (1000, 1)),
         rng.randint(0, 30000, (1000, 1))), axis=1).astype(theano.config.floatX)
    in_sub_neg_samples_init = rng.randint(0, 30000, (1000, 10))
    in_obj_neg_samples_init = rng.randint(0, 30000, (1000, 10))

    in_triples_minibatch = T.cast(theano.shared(in_triples_init, name="in_triples"), "int32")
    in_sub_neg_samples_minibatch = T.cast(theano.shared(in_sub_neg_samples_init, name="in_sub_neg_samples"), "int32")
    in_obj_neg_samples_minibatch = T.cast(theano.shared(in_obj_neg_samples_init, name="in_obj_neg_samples"), "int32")

    l = dist_mult_model.loss_on_set(in_triples, in_sub_neg_samples_mat, in_obj_neg_samples_mat)
    l2 = dist_mult_model.loss_on_set2(in_triples, in_sub_neg_samples_mat, in_obj_neg_samples_mat)

    cost = dist_mult_model.total_cost(in_triples, in_sub_neg_samples_mat, in_obj_neg_samples_mat)
    start = time.time()
    test_model = theano.function(
        inputs=[],
        outputs=[l2, cost],
        givens={
            in_triples: in_triples_minibatch,
            in_sub_neg_samples_mat: in_sub_neg_samples_minibatch,
            in_obj_neg_samples_mat: in_obj_neg_samples_minibatch
        }
    )
    end = time.time()
    print("Time taken to compile test = %f s" % (end - start))

    start = time.time()
    print("dummy DistMult model cost: %s" % test_model())
    end = time.time()
    print("Time taken to run test = %f s" % (end - start))

    pred_ranks = dist_mult_model.pred_ranks(in_triples)
    eval_model = theano.function(
        inputs=[],
        outputs=pred_ranks,
        givens={
            in_triples: in_triples_init[:284].astype(dtype=np.int32)
        }
    )
    ranks = eval_model()
    k = 15000
    print( "dummy mrr and hits@%d for 30000 entities = (%f, %f)" % ( k, np.mean(1 / ranks), float(np.nonzero(ranks <= k)[0].shape[0]) / float(ranks.shape[0]) ) )

