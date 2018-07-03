import numpy as np
import pickle, string, plac, collections, operator
import process_data

def scoring_function_v1(word_embeddings, pairs_id):
    """
    Computing the similarity score of two input sentences by averaging the similarity scores of each candidate pair.
    :param word_embeddings: a file contains all word vectors of the words in the vocabulary, one word embedding per line.
    :param pairs_id: a list of all uni-gram pairs from the input sequences.
    :return: the overall similarity score of two input sentences.
    """
    word_embeddings = np.array(word_embeddings)
    candidate_pairs, candidate_scores = get_candidate_pairs_v1(word_embeddings, pairs_id)
    overall_score = reduce(lambda x, y: x + y, candidate_scores) / len(candidate_scores)
    # print('the resulting pairs is {}'.format(candidate_pairs))
    # print('the resulting scores is {}'.format(candidate_scores))
    return overall_score

def get_candidate_pairs_v1(word_embeddings, pairs_id):
    """
    Building all candidates pairs and computing the corresponding similarity scores.
    :param word_embeddings: a file contains all word vectors of the words in the vocabulary, one word embedding per line.
    :param pairs_id: a list of all uni-gram pairs from the input sequences.
    :return: all candidates pairs and all similarity scores.
    """
    candidate_pairs = []
    candidate_scores = []
    score4pair = {}
    for each_id_pair in pairs_id:
        each_pair_score = process_data.compute_each_score(word_embeddings, each_id_pair)
        score4pair[tuple(each_id_pair)] = each_pair_score
    sorted_score4pair = sorted(score4pair.items(), key=operator.itemgetter(1), reverse=True)
    for pairs, score in sorted_score4pair:
        findItem = False
        if len(candidate_pairs) == 0:
            candidate_pairs.append(pairs)
            candidate_scores.append(score)
        else:
            for re in candidate_pairs:
                if len(set(pairs).intersection(re)) != 0:
                    findItem = True
                    break
            if findItem == False:
                candidate_pairs.append(pairs)
                candidate_scores.append(score)
    return candidate_pairs, candidate_scores

def main():
    """
    This is a demo for Word2Sent-V1, measuring the similarity score of two given sentences.
    :return: the similarity score.
    """

    # Step 1: Choosing the vocabulary and the word embeddings file.
    # path4words ='/shared/data_WordSentenceVector/model_lawinsider_full/lawinsider_full_tagged_OnlyWord/words_beagle'
    # path4words ='/shared/data_WordSentenceVector/model_googlenews/SentenceVector/words_beagle'
    path4words ='/shared/data_WordSentenceVector/model_wiki_glove/SentenceVector/words_glove'
    # path4emb = '/shared/data_WordSentenceVector/model_lawinsider_full/lawinsider_full_tagged_OnlyWord/vectors_beagle'
    # path4emb = '/shared/data_WordSentenceVector/model_googlenews/SentenceVector/vectors_beagle'
    path4emb = '/shared/data_WordSentenceVector/model_wiki_glove/SentenceVector/vectors_glove'

    # Step 2: Loading the vocabulary and the word embeddings file.
    print("loading words file from {}".format(path4words.split('/')[3]))
    words = pickle.load(open(path4words, 'rb'))
    print("loading word embeddings file from {}".format(path4emb.split('/')[3]))
    word_embeddings = pickle.load(open(path4emb, 'rb'))

    sentence1 = 'Three kids are sitting in the leaves.'
    sentence2 = 'Children in red shirts are playing with leaves.'
    # Step 3: Preprocessing the input sentence, removing the punctuation marks.
    sentence1 = sentence1.translate(None, string.punctuation)
    sentence2 = sentence2.translate(None, string.punctuation)
    # Step 4: Converting the two input sentences into sequences of index.
    seq1, seq2 = process_data.getSeqs(sentence1, sentence2, words)
    id1, m1 = process_data.prepare_data(seq1)
    id2, m2 = process_data.prepare_data(seq2)
    uni_pairs_id = process_data.unigram_pairs(id1, id2)
    # Step 5: Computing the similarity score.
    similarity_score = scoring_function_v1(word_embeddings, uni_pairs_id)
    print('the overall score is {}'.format(similarity_score))

if __name__ == '__main__':
    plac.call(main)
