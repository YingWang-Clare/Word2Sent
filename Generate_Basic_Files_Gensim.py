from django.utils.encoding import smart_str
from gensim.models import Word2Vec, KeyedVectors
from decimal import Decimal
import plac
import pickle
import os
import process_data

@plac.annotations(
    gensim_model_path=("Location of gensim's .bin file"),
    out_dir=("Location of output directory"),
)

def main(gensim_model_path, out_dir):
    """
    This function is used to generate words file, word embeddings file, and weight4ind file from the word embeddings in the gensim output format.
    """
    gensim_model = Word2Vec.load(gensim_model_path)
    words = {}
    n = 0
    vectors = []
    weightfile_name = os.path.join(out_dir, "weightfile.txt")
    weightfile = open(weightfile_name, "w")
    for string in gensim_model.wv.vocab:
        vocab = gensim_model.wv.vocab[string]
        freq, idx = vocab.count, vocab.index
        weightfile.write(smart_str(string))
        weightfile.write(" ")
        weightfile.write(smart_str(freq))
        weightfile.write("\n")
        vector = gensim_model.wv.syn0[idx]
        vectors.append(vector)
        words[string] = n
        n = n + 1

    vector_file = open(os.path.join(out_dir, "vectors"), "w")
    pickle.dump(vectors, vector_file)

    words_file = open(os.path.join(out_dir, "words"), "w")
    pickle.dump(words, words_file)

    weightpara = [1e-2, 1e-3, 1e-4]
    for a in weightpara:
        print("calculating word2weight with a = {}.".format(a))
        word2weight = process_data.getWordWeight(weightfile_name, a)
        print("calculating weight4ind with a = {}.".format(a))
        weight4ind = process_data.getWeight(words, word2weight)
        weight4ind_file = open(os.path.join(out_dir, "weight4ind_weightpara_%.E" % Decimal(a)), 'w')
        pickle.dump(weight4ind, weight4ind_file)

if __name__ == '__main__':
    plac.call(main)
