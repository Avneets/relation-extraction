import config
import logging
from structure import RelationTripleSet
import numpy as np

class Reader(object):

    def __init__(self):
        pass

    @staticmethod
    def read_data(filepath, entity_dict=None, relation_dict=None, has_count=False):
        if True:
            if entity_dict is None:
                entity_dict = dict()
            if relation_dict is None:
                relation_dict = dict()
            relation_triples_counter = dict()
            with open(filepath, 'r') as f:
                entity_index = 0
                relation_index = 0
                for line in f:
                    items = line.split()
                    if len(items) >= 3:
                        if items[0] not in entity_dict:
                            entity_dict[items[0]] = entity_index
                            entity_index += 1
                        if items[1] not in relation_dict:
                            relation_dict[items[1]] = relation_index
                            relation_index += 1
                        if items[2] not in entity_dict:
                            entity_dict[items[2]] = entity_index
                            entity_index += 1

                        relation_triple = (entity_dict[items[0]], relation_dict[items[1]], entity_dict[items[2]])
                        if relation_triple not in relation_triples_counter:
                            if has_count:
                                relation_triples_counter[relation_triple] = int(items[3])
                            else:
                                relation_triples_counter[relation_triple] = 1
                    else:
                        logging.debug("Some issue in - %s" % line)
            return RelationTripleSet(entity_dict, relation_dict, relation_triples_counter, has_count)
        # except:
        #     logging.error("Unexpected Error!")
        #     return None

if __name__ == "__main__":
    logging.info("Reading data from text relations")
    T = Reader.read_data(config.textRelationsFile, has_count=True)
    logging.info("\nNumber of Entitites = %d\nNumber of KB Relations = %d\nNumber of Tuples = %d" %
                 (len(T.entity_index), len(T.relation_index), len(T.relation_triples_counter)))
    logging.info("Reading data from KB")
    T = Reader.read_data(config.KBTrainFile)
    logging.info("\nNumber of Entitites = %d\nNumber of KB Relations = %d\nNumber of Tuples = %d" %
                 (len(T.entity_index), len(T.relation_index), len(T.relation_triples_counter)))
    logging.info("Testing negative sampling")
    index = np.random.randint(0, len(T.relation_triples_counter))
    s, r, o = T.relation_triples[index]
    logging.info("Negative sampling for (in object position) triple - (%s, %s, %s)" %
                 (T.reverse_entity_index[s], T.reverse_relation_index[r], T.reverse_entity_index[o]))
    negative_samples = T.sample_negative_instances((s, r, o), 10, False)
    logging.info("%s" % str([T.reverse_entity_index[e] for e in negative_samples]))