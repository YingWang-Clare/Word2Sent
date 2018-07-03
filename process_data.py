from __future__ import print_function
import pickle
import numpy as np
import codecs

def getSeqs(p1,p2,words):
    """
    :param p1: the first sentence in a String type.
    :param p2: the second sentence in a String type.
    :param words: a file contains all words in the corpus, one word per line.
    :return: two sequences of index for the two input sentences.
    """
    p1 = p1.split()
    p2 = p2.split()
    X1 = []
    X2 = []
    for i in p1:
        X1.append(lookupIDX(words,i))
    for i in p2:
        X2.append(lookupIDX(words,i))
    # print("X1:{}".format(X1))
    # print("X2:{}".format(X2))
    return X1, X2

def lookupIDX(words,w):
    """
    :param words: a file contains all words in the corpus, one word per line.
    :param w: a word.
    :return: the location (index) of the input word in the vocabulary (words file).
    """
    w = w.lower()
    if len(w) > 1 and w[0] == '#':
        w = w.replace("#","")
    if w in words:
        return words[w]
    elif 'UUUNKKK' in words:
        return words['UUUNKKK']
    else:
        return len(words) - 1
    
def prepare_data(seq):
    """
    :param seq: a sequence of index.
    :return: x is a numpy array of the sequence.
             x_mask is a numpy array of all 1 with the same length of the input sequence.
    """
    x = np.zeros(len(seq)).astype('int32')
    x_mask = np.zeros(len(seq)).astype('float32')
    x[:len(seq)] = seq
    x_mask[:len(seq)] = 1.
    x_mask = np.asarray(x_mask, dtype='float32')
    # print("x is {}".format(x))
    # print("x_mask is {}.".format(x_mask))
    return x, x_mask

def seq2weight(seq, mask, weight4ind):
    """
    :param seq: a sequence of index.
    :param mask: a numpy array indicates the real entries of the sequence.
    :param weight4ind: a file indicates the weight of the word according to the index of the word.
    :return: a sequence of weights of the input index sequence.
    """
    weight = np.zeros(seq.shape).astype('float32')
    for j in xrange(len(seq)):
        if mask[j] > 0 and seq[j] >= 0:
            weight[j] = weight4ind[seq[j]]
    weight = np.asarray(weight, dtype='float32')
    # print("w is {}".format(weight))
    return weight

def unigram_pairs(id1, id2):
    """
    Generating all uni-gram pairs of two input sentences.
    :param id1: a sequence of index from sentence 1.
    :param id2: a sequence of index from sentence 2.
    :return: a list of all uni-gram pairs from the input sequences.
    """
    uni_pairs = []
    for word1 in id1.tolist():
        for word2 in id2.tolist():
            uni_pairs.append([word1, word2])
    # print("uni_pair is {}".format(uni_pairs))
    # print("the # of uni pair is {}".format(len(uni_pairs)))
    return uni_pairs

def compute_each_score(word_embeddings, each_id_pair): # without weighting scheme
    """
    Computing the similarity score of the input pair based on the cosine distance, without the weighting scheme.
    :param word_embeddings: a file contains all word vectors of the words in the vocabulary, one word embedding per line.
    :param each_id_pair: a uni-gram pair of indices of two words.
    :return: a float number of the score.
    """
    emb1 = word_embeddings[each_id_pair[0], :]
    emb2 = word_embeddings[each_id_pair[1], :]
    inn = np.inner(emb1, emb2)
    # print('inner product is {}'.format(inn))
    emb1norm = np.sqrt(np.inner(emb1, emb1))
    # print('emb1norm is {}'.format(emb1norm))
    emb2norm = np.sqrt(np.inner(emb2, emb2))
    # print('emb2norm is {}'.format(emb2norm))
    each_pair_score = inn / emb1norm / emb2norm
    # print('each score is {}\n'.format(each_pair_score))
    return each_pair_score

def compute_each_score_weighted(word_embeddings, each_id_pair, each_weight_pair):
    """
    Computing the similarity score of the input pair based on the cosine distance, with the SIF weighting scheme.
    :param word_embeddings: a file contains all word vectors of the words in the vocabulary, one word embedding per line.
    :param each_id_pair: a uni-gram pair of indices of two words.
    :param each_weight_pair: a uni-gram pair of weights of two words.
    :return: a float number of the score.
    """
    emb1 = word_embeddings[each_id_pair[0], :]
    emb2 = word_embeddings[each_id_pair[1], :]
    inn = np.inner(emb1, emb2)
    # print('inner product is {}'.format(inn))
    emb1norm = np.sqrt(np.inner(emb1, emb1))
    # print('emb1norm is {}'.format(emb1norm))
    emb2norm = np.sqrt(np.inner(emb2, emb2))
    # print('emb2norm is {}'.format(emb2norm))
    each_pair_score = inn / emb1norm / emb2norm
    each_pair_score = ((each_weight_pair[0] + each_weight_pair[1]) / 2) * each_pair_score
    # print('each score is {}\n'.format(each_pair_score))
    return each_pair_score

def getWordmap(textfile):
    words = {}
    We = []
    f = open(textfile, 'r')
    lines = f.readlines()
    for (n, i) in enumerate(lines):
        i = i.split()
        j = 1
        standard = 301
        if len(i) != standard:
            continue
        v = []
        while j < len(i):
            v.append(float(i[j]))
            j += 1
        words[i[0]] = n
        v = np.array(v)
        We.append(v)
        if v.shape[0] != (standard - 1):
    We = np.array(We)
    print('type of we', type(We))
    return (words, We)

def getWordWeight(weightfile, a=1e-3):
    if a <=0: # when the parameter makes no sense, use unweighted
        a = 1.0
    word2weight = {}
    f = codecs.open(weightfile, "r", "utf-8")
    lines = f.readlines()
    N = 0
    for i in lines:
        i=i.strip()
        if(len(i) > 0):
            i=i.split()
            if(len(i) == 2):
                word2weight[i[0]] = float(i[1])
                N += float(i[1])
            else:
                print(i)
    for key, value in word2weight.iteritems():
        word2weight[key] = a / (a + value/N)
    return word2weight

def getWeight(words, word2weight):
    weight4ind = {}
    for word, ind in words.iteritems():
        if word in word2weight:
            weight4ind[ind] = word2weight[word]
        else:
            weight4ind[ind] = 1.0
    return weight4ind