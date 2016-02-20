import logging
from kbc.structure import KB
from kbc.utils import config


class KBReader(object):

    def __init__(self):
        pass

    @staticmethod
    def read_data(filepath, entity_dict=None, entity_pair_dict=None, relation_dict=None, has_count=False, add_new=True):
        if True:
            if entity_dict is None:
                entity_dict = dict()
            if relation_dict is None:
                relation_dict = dict()
            if entity_pair_dict is None:
                entity_pair_dict = dict()
            triples_counter = dict()
            with open(filepath, 'r') as f:
                entity_index = 0
                relation_index = 0
                entity_pair_index = 0
                for line in f:
                    items = line.split()
                    if len(items) >= 3:
                        if items[0] not in entity_dict:
                            if add_new:
                                entity_dict[items[0]] = entity_index
                                entity_index += 1
                            else:
                                continue
                        if items[1] not in relation_dict:
                            if add_new:
                                relation_dict[items[1]] = relation_index
                                relation_index += 1
                            else:
                                continue
                        if items[2] not in entity_dict:
                            if add_new:
                                entity_dict[items[2]] = entity_index
                                entity_index += 1
                            else:
                                continue
                        entity_pair = (entity_dict[items[0]], entity_dict[items[2]])
                        if entity_pair not in entity_pair_dict:
                            if add_new:
                                entity_pair_dict[entity_pair] = entity_pair_index
                                entity_pair_index += 1
                        #     else:
                        #         continue

                        relation_triple = (entity_dict[items[0]], relation_dict[items[1]], entity_dict[items[2]])
                        if relation_triple not in triples_counter:
                            if has_count:
                                triples_counter[relation_triple] = int(items[3])
                            else:
                                triples_counter[relation_triple] = 1
                    else:
                        logging.debug("Some issue in - %s" % line)
            return KB(entity_dict, relation_dict, entity_pair_dict, triples_counter, has_count)

if __name__ == "__main__":
    train_dataset = KBReader.read_data(config.KBTrainFile)
    valid_dataset = KBReader.read_data(filepath=config.KBValidationFile,
                                       entity_dict=train_dataset.entity_index,
                                       relation_dict=train_dataset.relation_index,
                                       add_new=True
                                       )
    test_dataset = KBReader.read_data(filepath=config.KBTestFile,
                                      entity_dict=train_dataset.entity_index,
                                      entity_pair_dict=train_dataset.entity_pair_index,
                                      relation_dict=train_dataset.relation_index,
                                      add_new=False
                                      )
    train_dataset.print_set_statistics()
    valid_dataset.print_set_statistics()
    test_dataset.print_set_statistics()
