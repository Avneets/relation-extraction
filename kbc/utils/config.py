import os
import logging
import theano
import numpy as np

# Data paths
rootDir = "/Users/pavankumar/Developer/NLPResearch/embeddings/toutanova"
dataPath = os.path.join(rootDir, "Release")
KBTrainFile = os.path.join(dataPath, "train.txt")
KBValidationFile = os.path.join(dataPath, "valid.txt")
KBTestFile = os.path.join(dataPath, "test.txt")
textRelationsFile = os.path.join(dataPath, "text_emnlp.txt")

saveModelsRoot = os.path.join(rootDir, "savedModels")

# Logging info
logging.basicConfig(level=logging.DEBUG)

# Theano settings
if not hasattr(theano.config, "floatX"):
    logging.info("floatX not found in config. Default to float32")
    theano.config.floatX = np.float32
floatX = theano.config.floatX
intX = 'int64'

# Training and Optimization params
ENTITY_EMBEDDINGS = "ent_emb"
RELATION_EMBEDDINGS = "rel_emb"
