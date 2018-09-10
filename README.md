# JointReps
JointReps is a joint model for learning distributed word vector representations (word embeddings) from both large text corpora and knowledge bases (KBs). 

This tool is the source codes for the proposed method reported in the published papers titled 
* "Joint Word Representation Learning Using a Corpus and a Semantic Lexicon" in AAAI-2016
* "[Jointly learning word embeddings using a corpus and a knowledge base" in PlosOne](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0193094)

# Contents
* ./src/reps.cc is the source code for training the model
* ./src/comb_NNE.py is the source code for NNE algorithm (Journal version)
* ./src/comb_MNE.py is the source code for MNE algorithm (Journal version)
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
   * this will call:
   * ./reps --dim=300 --epohs=20 --model=../work/model --alpha=0.01 --lmda=10000 --edges=../work/edges.sample --pairs=../work/synonyms
   * dim: word vectors dimension
   * epohs: number of iterations
   * alpha: initial learning rate of AdaGrad
   * lmda: the regularization coefficient in the proposed model
   * edges: co-occurrence matrix file
   * pairs: the lexicon file
* To combing the co-occurrence matrix file and the lexicon file, i.e. to expand the lexicon file with either NNE or MNE algorithms, with synonym relation type as an example
  * Go the ./src directory
  * Type python comb_NNE.py synonyms sampleEdges OutputFile
  * OutputFile: your expanded synonyms lexicon file
