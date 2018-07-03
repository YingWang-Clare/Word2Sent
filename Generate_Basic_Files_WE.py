import pickle
import sys
import os
from decimal import Decimal
import plac
sys.path.append('../src')
import data_io
import sim_algo
import eval
import params
import numpy as np
import process_data

@plac.annotations(
    word_embeddings_path=("Location of pre-trained word embeddings .txt file"),
    word_weight_path=("Location of the word weights .txt file"),
    out_dir=("Location of output directory"),
)

def main(word_embeddings_path, word_weight_path, out_dir):
    """
    This function is used to generate words file, word embeddings file, and weight4ind file from the pretrained word embeddings file, such as fastText or GloVe.
    """
    wordfile = word_embeddings_path
    weightfile = word_weight_path
    weightparas = [1e-2, 1e-3, 1e-4]
    (words, vectors) = process_data.getWordmap(wordfile)
    vector_file = open(os.path.join(out_dir, "vectors"), "w")
    pickle.dump(vectors, vector_file)
    words_file = open(os.path.join(out_dir, "words"), "w")
    pickle.dump(words, words_file)
    for weightpara in weightparas:
        print("calculating word2weight with a = {}.".format(weightpara))
        word2weight = process_data.getWordWeight(weightfile, weightpara)
        print("calculating weight4ind with a = {}.".format(weightpara))
        weight4ind = process_data.getWeight(words, word2weight)
        weight4ind_file = open(os.path.join(out_dir, "weight4ind_weightpara_%.E" % Decimal(weightpara)), 'w')
        pickle.dump(weight4ind, weight4ind_file)

if __name__ == '__main__':
    plac.call(main)
