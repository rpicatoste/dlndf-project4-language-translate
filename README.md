# Deep Learning Nanodegree Foundations - Project 4 - Language translation

The notebook can be run directly and will do the whole process.

An html version is stored with the result of the final code and hyperparameters selected.

Response to the 1st reviewers comments:

The hyperparameters have been adjusted to the problem. Certainly the embedding size needs to be lower, specially since it is a technique to reduce the dimensionality of a problem, and in this case it is low since the vocabulary is limited. 

The function sentence_to_seq has been corrected including the .lower() in the words before searching them in the dictionary.

Response to the 2nd reviewer comments:
I applied the suggested changes to the rnn size and layers, obtaining a very high accuracy. About the embedding size I did not increase it too much, since as pointed out by the 1st reviewer, given the small set of words, having it too high is counter productive. 

I tried translating phrases from the set given and those are properly translated. Then I changed words in those phrases, and sometimes the networks gives a good translation, but sometimes not. I think it's normal given the small corpus for training.