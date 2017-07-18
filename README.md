# Deep Learning Nanodegree Foundations - Project 4 - Language translation

The notebook can be run directly and will do the whole process.

An html version is stored with the result of the final code and hyperparameters selected.

Response to the reviewers comments:
The hyperparameters have been adjusted to the problem. Certainly the embedding size needs to be lower, specially since it is a technique to reduce the dimensionality of a problem, and in this case it is low since the vocabulary is limited. 

The function sentence_to_seq has been corrected including the .lower() in the words before searching them in the dictionary.
