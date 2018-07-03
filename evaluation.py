import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle, time, plac, string, collections, operator
import os
from os import listdir
from os.path import isfile, join
from word2sent_v1 import scoring_function_v1
from word2sent_v2 import scoring_function_v2
from word2sent_v3 import scoring_function_v3
import process_data

def get_correlation(word_embeddings, words, test_file, scoring_function):
    """
    Testing the model on the given test file and producing the evaluation results with the Pearson's Coefficent and MSE.
    :param word_embeddings: a file contains all word vectors of the words in the vocabulary, one word embedding per line.
    :param words: a file contains all words in the corpus, one word per line.
    :param test_file: a test dataset.
    :param scoring_function: no weighting scheme scoring function (V1).
    :return: the Pearson's Coefficent and MSE
    """
    print("evaluate with the model {}\n".format(str(scoring_function).split('_')[2]))
    test_file = open(test_file)
    lines = test_file.readlines()
    golds = []
    preds = []
    index = []
    idx = 0
    for num, i in enumerate(lines):
        if num % 100 == 0:
            print('{} lines have been proceeded.\n'.format(num))
        i = i.split("\t")
        sent1 = i[0].translate(None, string.punctuation)
        sent2 = i[1].translate(None, string.punctuation)
        score = float(i[2])
        seq1, seq2 = process_data.getSeqs(sent1, sent2, words)
        id1, m1 = process_data.prepare_data(seq1)
        id2, m2 = process_data.prepare_data(seq2)
        uni_pairs_id = process_data.unigram_pairs(id1, id2)
        sentence_score = scoring_function(word_embeddings, uni_pairs_id)
        golds.append(score)
        preds.append(sentence_score)
        index.append(idx)
        idx += 1
    golds = np.asarray(golds)
    preds = np.asarray(preds)
    MSE = sqrt(mean_squared_error(golds, preds))
    return pearsonr(preds, golds)[0], MSE

def get_correlation_weighted(word_embeddings, words, weight4ind, test_file, scoring_function):
    """
    Testing the model on the given test file and producing the evaluation results with the Pearson's Coefficent and MSE.
    :param word_embeddings: a file contains all word vectors of the words in the vocabulary, one word embedding per line.
    :param words: a file contains all words in the corpus, one word per line.
    :param weight4ind: a file indicates the weight of the word according to the index of the word.
    :param test_file: a test dataset.
    :param scoring_function: with weighting scheme scoring function (V2 & V3).
    :return: the Pearson's Coefficent and MSE
    """
    print("evaluate with the model {}\n".format(str(scoring_function).split('_')[2]))
    test_file = open(test_file)
    lines = test_file.readlines()
    golds = []
    preds = []
    index = []
    idx = 0
    for num, i in enumerate(lines):
        if num % 100 == 0:
            print('{} lines have been proceeded.\n'.format(num))
        i = i.split("\t")
        sent1 = i[0].translate(None, string.punctuation)
        sent2 = i[1].translate(None, string.punctuation)
        score = float(i[2])
        seq1, seq2 = process_data.getSeqs(sent1, sent2, words)
        id1, m1 = process_data.prepare_data(seq1)
        id2, m2 = process_data.prepare_data(seq2)
        weight1 = process_data.seq2weight(id1, m1, weight4ind)
        weight2 = process_data.seq2weight(id2, m2, weight4ind)
        uni_pairs_id = process_data.unigram_pairs(id1, id2)
        uni_pairs_weight = process_data.unigram_pairs(weight1, weight2)
        sentence_score = scoring_function(word_embeddings, uni_pairs_id, uni_pairs_weight)
        golds.append(score)
        preds.append(sentence_score)
        index.append(idx)
        idx += 1
    golds = np.asarray(golds)
    preds = np.asarray(preds)
    MSE = sqrt(mean_squared_error(golds, preds))
    return pearsonr(preds, golds)[0], MSE


def main():
    print('evaluation starts.')
    # Step 1: Choosing the vocabulary and the word embeddings file.
    path4words = '/shared/data_WordSentenceVector/model_lawinsider_full/lawinsider_full_tagged_OnlyWord/words_beagle'
    # path4words = '/shared/data_WordSentenceVector/model_wiki_glove/SentenceVector/words_glove'
    # path4words = '/shared/data_WordSentenceVector/model_fasttext_cc/SentenceVector/words_ft_cc'
    path4emb = '/shared/data_WordSentenceVector/model_lawinsider_full/lawinsider_full_tagged_OnlyWord/vectors_beagle'
    # path4emb = '/shared/data_WordSentenceVector/model_wiki_glove/SentenceVector/vectors_glove'
    # path4emb = '/shared/data_WordSentenceVector/model_fasttext_cc/SentenceVector/vectors_ft_cc'
    path4weight = '/shared/data_WordSentenceVector/model_lawinsider_full/lawinsider_full_tagged_OnlyWord/weight4ind_weightpara_1E-03'
    # path4weight = '/shared/data_WordSentenceVector/model_wiki_glove/SentenceVector/weight4ind_glove_1e-03'
    # path4weight = '/shared/data_WordSentenceVector/model_fasttext_cc/SentenceVector/weight4ind_weightpara_1E-03'

    # Step 2: Loading the vocabulary and the word embeddings file.
    print("loading words file from {}".format(path4words.split('/')[3]))
    words = pickle.load(open(path4words, 'rb'))
    print("loading word embeddings file from {}".format(path4emb.split('/')[3]))
    word_embeddings = pickle.load(open(path4emb, 'rb'))
    print("loading weight4ind file from {}".format(path4weight.split('/')[3]))
    weight4ind = pickle.load(open(path4weight, 'rb'))

    # Step 3: Creating a dictionary to find scoring function and its string
    func_dictionary = {'scoring_function_v1': scoring_function_v1,
                       'scoring_function_v2': scoring_function_v2,
                       'scoring_function_v3': scoring_function_v3,
                       }
    # Step 4: Adding all test files into a list
    test_folder = './eval_data/'
    test_list = [join(test_folder, f) for f in listdir(test_folder) if isfile(join(test_folder, f))]

    # Step 5: Preparing to write the processing information and the evaluation results to a .txt file.
    f = open("overall_result.txt", 'w')
    f.write("loading words file from {}\n".format(path4words.split('/')[3]))
    f.write("loading word embeddings file from {}\n".format(path4emb.split('/')[3]))
    f.write("loading weight4ind file from {}\n".format(path4weight.split('/')[3]))
    for tf in test_list:
        f.write("test file is {}\n".format(str(tf)))
        sf = 'scoring_function_v1'
        pearson, MSE = get_correlation(word_embeddings, words, tf, func_dictionary[sf])
        f.write("evaluate with the model {}\n".format(str(sf).split('_')[2]))
        f.write('pearson is {} and MSE is {}\n'.format(pearson, MSE))
        for i in (7,8):
            sf = 'scoring_function_v' + str(i)
            pearson, MSE = get_correlation_weighted(word_embeddings, words, weight4ind, tf, func_dictionary[sf])
            f.write("evaluate with the model {}\n".format(str(sf).split('_')[2]))
            f.write('pearson is {} and MSE is {}\n'.format(pearson, MSE))

if __name__ == '__main__':
    plac.call(main)
