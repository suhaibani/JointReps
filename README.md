# JointReps
JointReps is a joint model for learning distributed word vector representations (word embeddings) from both large text corpora and knowledge bases (KBs). 

This tool is the source codes for the proposed method reported in the papers titled "Joint Word Representation Learning Using a Corpus and a Semantic Lexicon" and its extened version "Jointly Learning Word Embeddings using a Corpus and a Knowledge Base" submitted to AAAI-2016 and IEEE-TDKE respectively.

# Contents
* ./src/reps.cc is the source code for training the model
* ./src/comb_NNE.py is the source code for expanding the knowledge base with NNE algorithm (IEEE-TDKE version)
* ./src/comb_MNE.py is the source code for expanding the knowledge base with MNE algorithm (IEEE-TDKE version)
* ./work includes all the lexicon files and a small co-occurrence matrix sample (sampleEdges)
* ./vectors the pretrained word vectors are available for download

# Requirements
* The model ./src/reps.cc is written in C++, therefore a C++0x compiler is required
* [C++ Eigen Library](http://eigen.tuxfamily.org/index.php?title=Main_Page) need to be installed
  * Alternatively, instead of installing Eigen, you can simply create a directory named 'eigen' in the same level as ./src and copy the source code of Eigen into it
* [Numpy](http://www.numpy.org/) library for ./work/comb_NNE.py and comb_MNE.py expansion algorithms

# Installation
Move to the ./src directory and type make

# Examples
* To train the model with a synonym file as an example, a small co-occurrence matrix ./work/sampleEdges and ./src/Makefile are provided
 * To run it, type make run
