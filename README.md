# JointReps
JointReps is a joint model for learning distributed word vector representations (word embeddings) from both large text corpora and knowledge bases (KBs). 

This tool is the source codes for the proposed method reported in the papers titled "Joint Word Representation Learning Using a Corpus and a Semantic Lexicon" and its extened version "Jointly Learning Word Embeddings using a Corpus and a Knowledge Base" submitted to AAAI-2016 and IEEE-TDKE respectively.

# Contents
* ./src/reps.cc is the source code for training the model
* ./src/comb_NNE.py is the source code for expanding the knowledge base with NNE algorithm (IEEE-TDKE version)
* ./src/comb_MNE.py is the source code for expanding the knowledge base with MNE algorithm (IEEE-TDKE version)
* ./work/ includes all the lexicon files
* ./vectors/ the pretrained word vectors are available for download

# Requirements
