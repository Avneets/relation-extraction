import os
import logging

dataPath = "../Release"
KBTrainFile = os.path.join(dataPath, "train.txt")
KBValidationFile = os.path.join(dataPath, "valid.txt")
KBTestFile = os.path.join(dataPath, "test.txt")
textRelationsFile = os.path.join(dataPath, "text_emnlp.txt")
logging.basicConfig(level=logging.DEBUG)
