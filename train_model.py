"""This module trains the binary classifier on the efftreat data."""

import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import func
from word_embedding import Embeddings

import pdb

version = 'V2p0p0'
embedding_choice = 'BoW'

# Whether to use a saved word embedding or not.
USE_SAVED_EMBEDDING = True

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


    embedding = Embeddings(df_clean['comments_clean'].values,
                           word_in_vocab,
                           comments_length)

    if embedding_choice == 'one_hot':
    # try simple word embedding with one-hot encoding with average over the sentence.
    # So one comment is represented by a sparse vector with some weight != 0
    # corresponding to the mean values.
        embedded_comments, labels = embedding.one_hot(average_over = True,
                                                      save = True,
                                                      labels = df_clean['labels'].values)
    if embedding_choice == 'BoW':
    # Try the bag of words encoding.
        embedded_comments, labels = embedding.bag_of_word(save = True,
                                                          labels = df_clean['labels'].values,
                                                          mode = 'binary')

    if labels is None:
        labels = df_clean['labels'].values

elif USE_SAVED_EMBEDDING:

    if embedding_choice == 'one_hot':
        archive_encod = np.load('./tmp/sparse_matrix_one_hot.npz', 'rb', allow_pickle=True)

    if embedding_choice == 'BoW':
        archive_encod = np.load('./tmp/sparse_matrix_BoW.npz', 'rb', allow_pickle=True)

    embedded_comments = archive_encod['embed_array']
    labels = archive_encod['labels']

    Ntrain = int(0.25 * len(labels))


# First simple model
X_train_sparse = embedded_comments[Ntrain:]
X_test_sparse = embedded_comments[0:Ntrain]

X_train = np.array([X_train_sparse_i.toarray()[0].tolist() for X_train_sparse_i in X_train_sparse])
X_test = np.array([X_test_sparse_i.toarray()[0].tolist() for X_test_sparse_i in X_test_sparse])

Y_train = labels[Ntrain:]
Y_test = labels[0:Ntrain]

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(16, activation='relu', input_shape = (X_train.shape[1],)))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

BATCH_SIZE = 64
EPOCHS = 500
L_R = 0.01
sgd = tf.keras.optimizers.SGD(learning_rate=L_R)

history_one = func.compile_and_fit(model, BATCH_SIZE, EPOCHS, sgd, X_train, Y_train, X_test, Y_test)

with open('./histories/history_'+version, 'wb') as f:
    pickle.dump(history_one.history, f)

model_perf = model.evaluate(X_test, Y_test, verbose = False)
accuracy = int(np.round(model_perf[2], 2)*100.)
precision = int(np.round(model_perf[3], 2)*100.)
recall = int(np.round(model_perf[4], 2)*100.)

with open('./histories/model_'+version+'.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.write('\n')
    f.write('BATCH_SIZE:%i'%(BATCH_SIZE)+'\n')
    f.write('EPOCHS:%i'%(EPOCHS)+'\n')
    f.write('L_R:%f'%(L_R)+'\n')
    f.write('Embedding:%s'%(embedding_choice)+'\n')
    f.write('-------------- \n')
    f.write('This model performs with %i percent of accuracy,'%(accuracy)+\
      ' %i percent of precision,'%(precision)+\
      ' and %i percent of recall. \n'%(recall))
    f.write('--------------')
