import scipy
import numpy as np
import pandas as pd
from tqdm import tqdm

class Embeddings():
    """
    A class to perform word embedding.
    str_arr: pre-embedding sentences (e.g. the comments).
    encoding: matrix containing the vectors corresponding to every word in the vocab.

    return embed_arr an embedded matrix.
    """

    def __init__(self, str_arr, word_dict, max_length):
        self.str_arr = str_arr
        self.word_dict = word_dict
        self.max_length = max_length    
    def one_hot(self, average_over = False, save = False, labels = None):

        if save and (labels is None):
            raise AssertionError("Can't save encoding without providing labels. Otherwise \
                                  it's gonna be all shuffled one uploading the saved encoding!")

        # Convert the vocabulary into one-hot (averaged first to test)
        df_word_dict = pd.DataFrame(self.word_dict, columns = ['words'])
        one_hot_encod = pd.get_dummies(df_word_dict['words'])

        # Create sparse matrices of the one hot encoding
        one_hot_encod_sparse = scipy.sparse.csr_matrix(one_hot_encod.values)


        # We create embedded_array which contains the csr_matrix values
        # of the words in each of the comments.
        embedded_array = []
        with tqdm(total=len(self.str_arr)) as pbar:
            for str_arr_i in self.str_arr:

                embedded_sentence = []
                for word in str_arr_i.split(' '):

                    if word in self.word_dict:
                        embedded_sentence.append(one_hot_encod_sparse[self.word_dict.index(word)])
                    else:
                        pass

                    if len(embedded_sentence) >= self.max_length:
                        break

                if len(embedded_sentence) == 0:
                    embedded_sentence = [scipy.sparse.csr_matrix(np.zeros(len(self.word_dict)))]

                if average_over:
                    av_embedded_sentence = scipy.sparse.csr_matrix(np.zeros(len(self.word_dict)))
                    for embedded_word_i in embedded_sentence:
                        av_embedded_sentence += embedded_word_i
                    
                    # Create a vector of size that of the vocabulary vocab 
                    # which is averaged over the comment.
                    # Therefore, each comment is encoded by a single vector.
                    # The only non-zero values are the averaged values accross the sentence.
                    # More complex embedding can be tested later on.
                    embedded_array.append(av_embedded_sentence/len(embedded_sentence))

                else:

                    print('Not coded yet [embedding averaged over the com]')
                    pdb.set_trace()
                    # Need padding since not all the sentences are the same length.
                    # encod_com = embed_arr

                pbar.update()                    

        assert len(embedded_array) == len(self.str_arr),  "The length of the word embedding vectors"+\
                                                          " is not the same length as the comment."

        if save:
            assert len(embedded_array) == len(labels),  "The length of the word embedding vectors"+\
                                                        " is not the same length as the labels."

            with open('./tmp/sparse_matrix_one_hot.npz', 'wb') as file:
                np.savez_compressed(file, embed_array = embedded_array, labels = labels)

            return embedded_array, labels

        else:

            return embedded_array, labels
