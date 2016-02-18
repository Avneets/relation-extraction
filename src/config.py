import os
import logging
import theano
import numpy as np

# Data paths
dataPath = "../Release"
KBTrainFile = os.path.join(dataPath, "train.txt")
KBValidationFile = os.path.join(dataPath, "valid.txt")
KBTestFile = os.path.join(dataPath, "test.txt")
textRelationsFile = os.path.join(dataPath, "text_emnlp.txt")

# Logging info
logging.basicConfig(level=logging.DEBUG)

# Theano settings
if not hasattr(theano.config, "floatX"):
    logging.info("floatX not found in config. Default to float64")
    theano.config.floatX = np.float64

# Training and Optimization params
batchSize = 100
numNegSamples = 100
