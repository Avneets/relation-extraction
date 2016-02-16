import os
import logging

# Data paths
dataPath = "../Release"
KBTrainFile = os.path.join(dataPath, "train.txt")
KBValidationFile = os.path.join(dataPath, "valid.txt")
KBTestFile = os.path.join(dataPath, "test.txt")
textRelationsFile = os.path.join(dataPath, "text_emnlp.txt")

# Logging info
logging.basicConfig(level=logging.DEBUG)

# Training and Optimization params
batchSize = 100
numNegSamples = 100
