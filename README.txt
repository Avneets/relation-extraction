This project currently implements KB completion with modified models E and DistMult from "Relation Extraction with Matrix Factorization and Universal Schemas" by Sebastian Riedel et al., 2013 and "Embedding entities and relations for learning and inference in knowledge bases" by Yang et al., 2015. It will be extended to incorporate PPDB semantic similarity style objective into the current objective and perform relation extraction on sentential instances as opposed to KB completion.

Note for running the code:

This is a work in progress. Currently has working KB models (E, DistMult and hybrid E+DistMult). To run the current build modify the config variables KBTrainFile, KBValidationFile, KBTestFile to point to train, validation and test files consisting of KB triples in (subject, relation, object) format. You could download the "FB15K-237 Knowledge Base Completion Dataset" here - http://research.microsoft.com/en-us/people/kristout/.

Necessary dependencies - Theano, Lasagne

Provide the params in kb_main.py file and run
THEANO_FLAGS=exception_verbosity=high,optimizer=None,floatX=float32 python kb_main.py