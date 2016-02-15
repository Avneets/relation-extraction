import config
import logging


class Reader(object):

    def __init__(self):
        pass

    @staticmethod
    def read_data(filepath, entity_dict=None, relation_dict=None, has_count=False):
        try:
            if entity_dict is None:
                entity_dict = dict()
            if relation_dict is None:
                relation_dict = dict()
            relation_tuples = list()
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

                        if has_count:
                            assert len(items) == 4
                            relation_tuples.append((entity_dict[items[0]], relation_dict[items[1]], entity_dict[items[2]],
                                                int(items[3])))
                        else:
                            relation_tuples.append((entity_dict[items[0]], relation_dict[items[1]], entity_dict[items[2]]))
                    else:
                        logging.debug("Some issue in - %s" % line)


            return entity_dict, relation_dict, relation_tuples

        except:
            logging.ERROR("Unexpected Error!")
            return None

if __name__ == "__main__":
    logging.info("Reading data from KB")
    E, R, T = Reader.read_data(config.KBTrainFile)
    logging.info("\nNumber of Entitites = %d\nNumber of KB Relations = %d\nNumber of Tuples = %d\n" % (len(E), len(R), len(T)))
    logging.info("Reading data from text relations")
    E, R, T = Reader.read_data(config.textRelationsFile)
    logging.info("\nNumber of Entitites = %d\nNumber of KB Relations = %d\nNumber of Tuples = %d" % (len(E), len(R), len(T)))
