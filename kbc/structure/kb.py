import numpy as np
import theano

from kbc.utils import config


class KB(object):

    def __init__(self, entity_index, relation_index, entity_pair_index, relation_triples_counter, has_count):
        """
        Data structure for storing a set of relation triples of both KB and text relations

        :param entity_index: map of entity strings to a number
        :param relation_index: map of relation strings to a number
        :param relation_triples_counter: map of a relation triple to number of times
        it occurs (relevant only for text relations)
        :param has_count: has the count (True for text relations)
        """

        self.entity_index = entity_index
        self.relation_index = relation_index
        self.entity_pair_index = entity_pair_index
        self.triples_counter = relation_triples_counter
        self.triples = relation_triples_counter.keys()
        self.has_count = has_count
        self.rng = np.random

        self.reverse_entity_index = {value: key for key, value in self.entity_index.items()}
        self.reverse_relation_index = {value: key for key, value in self.relation_index.items()}
        self.reverse_entity_pair_index = {value: key for key, value in self.entity_pair_index.items()}

        self.n_entities = len(self.entity_index)
        self.n_relations = len(self.relation_index)
        self.n_pairs = len(self.entity_pair_index)
        self.n_triples = len(self.triples)

        self.batch_index = 0

    def sample_negative_instances(self, relation_triple, num_samples, for_subject):
        """
        Gives the samples for (?, r, e_o) and (e_s, r, ?)

        :param relation_triple: the positive triple for which negative samples are given
        :param num_samples: number of negative samples needed
        :param for_subject: True if samples are for (?, r, e_o)
        :return: list of indices indicating the negative samples
        """
        sub, rel, obj = relation_triple
        negative_instances = set()
        entity_indices = self.entity_index.values()
        while len(negative_instances) < num_samples:
            entity = entity_indices[self.rng.randint(0, len(entity_indices))]
            if entity in negative_instances:
                continue
            if for_subject and (entity, rel, obj) in self.triples_counter:
                continue
            elif not for_subject and (sub, rel, entity) in self.triples_counter:
                continue
            negative_instances.add(entity)
        return list(negative_instances)

    def print_set_statistics(self):
        print("Number of entities = %d" % self.n_entities)
        print("Number of relations = %d" % self.n_relations)
        print("Number of relation triples = %d" % self.n_triples)

    def construct_triple_string(self, relation_triple):
        """
        Helper function

        :param relation_triple: (e_s, r, e_o)
        :return: relation triple in the format e_s_string \t rel_string \t e_o_string
        """
        sub, rel, obj = relation_triple
        return self.reverse_entity_index[sub] + '\t' + self.reverse_relation_index[rel] + \
               '\t' + self.reverse_entity_index[obj]

    def generate_batch(self, batch_size):
        """

        :param batch_size:  size of the batch. -1 for all triples.
        :return: returns batch of triples of the min(batch_size, num_triples)
        """

        if batch_size < 0: batch_size = self.n_triples
        batch_size = min(batch_size, self.n_triples)
        triples = np.zeros((batch_size, 3)).astype(theano.config.floatX)
        for i in xrange(batch_size):
            i_range = (i + self.batch_index) % self.n_triples
            triples[i] = np.asarray(list(self.triples[i_range]), dtype=theano.config.floatX)
        self.batch_index = (self.batch_index + batch_size) % self.n_triples
        return (triples,)
