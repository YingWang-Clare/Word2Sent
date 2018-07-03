# Word2Sent

Word2Sent is a model for measuring the relatedness of two sentences. 
The model takes two sentences with arbitrary length as input and outputs a numerical value ranging from -1 to 1 as the degree of similarity. 
If the degree of similarity between two input sentences is very high, which means these two sentences are very similar to each other, 
then the output score would be near to 1; Otherwise, the value of the output would be close to -1.

Word2Sent uses pre-trained word embeddings derived by different word embedding models (i.e., Word2Vec, GloVe, or fastText)
on the unlabeled dataset, to represent words of the input sentence, and each word is represented by a fixed-dimension word vector. 

There are three different versions of Word2Sent, denoted as Word2Sent-V1, Word2Sent-V2, and Word2Sent-V3, respectively.
