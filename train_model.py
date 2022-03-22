"""This module trains the binary classifier on the efftreat data."""

import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import func
from word_embedding import Embeddings

# Whether to use a saved word embedding or not.
USE_SAVED_EMBEDDING = False

if USE_SAVED_EMBEDDING is not True:

    if os.path.exists('./data/efftreat_clean_label.csv') is not True:

        # Calls the function perform_cleaning which
        # cleans the data (see the function for more details).
        func.match_and_clean('./data/comments_long.csv',
                             './data/efftreat_long.csv',
                             'comment',
                             'efftreat',
                             './data/efftreat_clean_label.csv')

    # Load the dataframe with cleaned comments and corresponding labels
    df_clean = pd.read_csv('./data/efftreat_clean_label.csv')
    print('We have %i cleaned comments.'%(len(df_clean)))

    # Here, we have a cleaned sample of 114 909 comments regarding effective treatments
    # and their labels (either 1 or 0 for usefull or useless)
    # The aim is to use NLP to decide whether a comment is useful.

    # Shuffle the dataframe
    df_clean = df_clean.sample(frac=1).reset_index(drop=True)

    # We split into Train and Test samples (25%), and we pre-treat on the train sample only.
    Ntrain = int(0.25 * len(df_clean))
    df_clean_train = df_clean.filter(['comments_clean','labels'], axis=1)[Ntrain:]
    print('Of which %i are used for training and pre-treatment.'%(len(df_clean_train)))

    # Use pre-treatment of the data to extract the lengths
    # of the comments as well as the vocabulary to use.
    comments_length, word_in_vocab = func.perform_pretreat(df_clean_train)

    comments_embedded = Embeddings(df_clean['comments_clean'].values,
                                   word_in_vocab,
                                   comments_length)

    # try simple word embedding with one-hot encoding with average over the sentence.
    # So one comment is represented by a sparse vector with some weight != 0
    # corresponding to the mean values.
    embed_matrix_sparse, labels = comments_embedded.one_hot(average_over = True,
                                                            save = True,
                                                            labels = df_clean['labels'].values)

    if labels is None:
        labels = df_clean['labels'].values

elif USE_SAVED_EMBEDDING:

    archive_encod = np.load('./tmp/sparse_matrix_one_hot.npz', 'rb', allow_pickle=True)

    embed_matrix_sparse = archive_encod['embed_arr_full']
    labels = archive_encod['labels']

    Ntrain = int(0.25 * len(labels))

# First simple model
X_train_sparse = embed_matrix_sparse[Ntrain:]
X_test_sparse = embed_matrix_sparse[0:Ntrain]

X_train = np.array([X_train_sparse_i.toarray()[0].tolist() for X_train_sparse_i in X_train_sparse])
X_test = np.array([X_test_sparse_i.toarray()[0].tolist() for X_test_sparse_i in X_test_sparse])

Y_train = labels[Ntrain:]
Y_test = labels[0:Ntrain]

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(32, activation='relu', input_shape = (X_train.shape[1],)))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

BATCH_SIZE = 128
EPOCHS = 300
L_R = 0.01
sgd = tf.keras.optimizers.SGD(learning_rate=L_R)

history_one = func.compile_and_fit(model, BATCH_SIZE, EPOCHS, sgd, X_train, Y_train, X_test, Y_test)

with open('./histories/history_one_300_64_one_hot', 'wb') as f:
    pickle.dump(history_one.history, f)
