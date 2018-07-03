import numpy as np
import pickle, string, plac
import collections, operator
import process_data

def scoring_function_v2(word_embeddings, pairs_id, pairs_weight):
    """
    Computing the similarity score of two input sentences by taking the weighted average of the similarity scores of each candidate pair.
    :param word_embeddings: a file contains all word vectors of the words in the vocabulary, one word embedding per line.
    :param pairs_id: a list of all uni-gram pairs from the input sequences.
    :return: the overall similarity score of two input sentences.
    """
    word_embeddings = np.array(word_embeddings)
    candidate_pairs, candidate_scores, candidate_weights_pairs = get_candidate_pairs_v2(word_embeddings, pairs_id, pairs_weight)
    averaged_weights = [(weights[0] + weights[1]) / 2 for weights in candidate_weights_pairs]
    overall_score = np.average(candidate_scores, weights=averaged_weights)
    # print('the resulting pairs is {}'.format(candidate_pairs))
    # print('the resulting scores is {}'.format(candidate_scores))
    # print('the resulting weights is {}'.format(averaged_weights))
    return overall_score

def get_candidate_pairs_v2(word_embeddings, pairs_id, pairs_weight):
    """
    Building all candidates pairs and computing the corresponding similarity scores.
    :param word_embeddings: a file contains all word vectors of the words in the vocabulary, one word embedding per line.
    :param pairs_id: a list of all uni-gram pairs of index from the input sequences.
    :param pairs_weight: a list of all uni-gram pairs of weights from the input sequences.
    :return: all candidates pairs, all similarity scores, and all candidates weights
    """
    candidate_pairs = []
    candidate_scores = []
    candidate_weights = []
    score4pair = {}
    idx4weight = 0
    for each_id_pair, each_weight_pair in zip(pairs_id, pairs_weight):
        each_bondle = [each_id_pair[0], each_id_pair[1], each_weight_pair[0], each_weight_pair[1]]
        each_pair_score = process_data.compute_each_score(word_embeddings, each_id_pair)
        score4pair[tuple(each_bondle)] = each_pair_score
    sorted_score4pair = sorted(score4pair.items(), key=operator.itemgetter(1), reverse=True)
    for bondle, score in sorted_score4pair:
        findItem = False
        if len(candidate_pairs) == 0:
            pairs = [bondle[0], bondle[1]]
            candidate_pairs.append(pairs)
            candidate_scores.append(score)
            weights = [bondle[2], bondle[3]]
            candidate_weights.append(weights)
        else:
            for re in candidate_pairs:
                pairs = [bondle[0], bondle[1]]
                if len(set(pairs).intersection(re)) != 0:
                    findItem = True
                    break
            if findItem == False:
                pairs = [bondle[0], bondle[1]]
                candidate_pairs.append(pairs)
                candidate_scores.append(score)
                weights = [bondle[2], bondle[3]]
                candidate_weights.append(weights)
        idx4weight = idx4weight + 1
    return candidate_pairs, candidate_scores, candidate_weights

def main():
    """
    This is a demo for Word2Sent-V2, measuring the similarity score of two given sentences with the weighted scheme.
    :return: the similarity score.
    """
    # Step 1: Choosing the vocabulary and the word embeddings file.
    # path4words = '/shared/data_WordSentenceVector/model_lawinsider/SentenceVectorNoTagging/words_beagle'
    # path4words ='/shared/data_WordSentenceVector/model_lawinsider_full/lawinsider_full_tagged_OnlyWord/words_beagle'
    # path4words ='/shared/data_WordSentenceVector/model_googlenews/SentenceVector/words_beagle'
    path4words ='/shared/data_WordSentenceVector/model_wiki_glove/SentenceVector/words_glove'
    # path4emb = '/shared/data_WordSentenceVector/model_lawinsider/SentenceVectorNoTagging/vectors_beagle'
    # path4emb = '/shared/data_WordSentenceVector/model_lawinsider_full/lawinsider_full_tagged_OnlyWord/vectors_beagle'
    # path4emb = '/shared/data_WordSentenceVector/model_googlenews/SentenceVector/vectors_beagle'
    path4emb = '/shared/data_WordSentenceVector/model_wiki_glove/SentenceVector/vectors_glove'
    # path4weight = '/shared/data_WordSentenceVector/model_lawinsider/SentenceVectorNoTagging/weight4ind_weightpara_1e-03'
    # path4weight = '/shared/data_WordSentenceVector/model_lawinsider_full/lawinsider_full_tagged_OnlyWord/weight4ind_weightpara_1E-03'
    # path4weight = '/shared/data_WordSentenceVector/model_googlenews/SentenceVector/weight4ind_weightpara_1e-03'
    path4weight = '/shared/data_WordSentenceVector/model_wiki_glove/SentenceVector/weight4ind_glove_1e-03'
    
    # Step 2: Loading the vocabulary and the word embeddings file.
    print("loading words file from {}".format(path4words.split('/')[3]))
    words = pickle.load(open(path4words, 'rb'))
    print("loading word embeddings file from {}".format(path4emb.split('/')[3]))
    word_embeddings = pickle.load(open(path4emb, 'rb'))
    print("loading weight4ind file from {}".format(path4weight.split('/')[3]))
    weight4ind = pickle.load(open(path4weight, 'rb'))

    sentence2 = 'three kids are sitting in the leaves'
    sentence1 = 'children in red shirts are playing with leaves'
    # Step 3: Preprocessing the input sentence, removing the punctuation marks.
    sentence1 = sentence1.translate(None, string.punctuation)
    sentence2 = sentence2.translate(None, string.punctuation)
    # Step 4: Converting the two input sentences into sequences of index.
    seq1, seq2 = process_data.getSeqs(sentence1, sentence2, words)
    id1, m1 = process_data.prepare_data(seq1)
    id2, m2 = process_data.prepare_data(seq2)
    weight1 = process_data.seq2weight(id1, m1, weight4ind)
    weight2 = process_data.seq2weight(id2, m2, weight4ind)  # w is np.ndarray
    uni_pairs_weight = process_data.unigram_pairs(weight1, weight2)  # list of list
    uni_pairs_id = process_data.unigram_pairs(id1, id2)
    # Step 5: Computing the similarity score.
    similarity_score = process_data.scoring_function_v2(word_embeddings, uni_pairs_id, uni_pairs_weight)
    print('the overall score is {}'.format(similarity_score))

if __name__ == '__main__':
    plac.call(main)
