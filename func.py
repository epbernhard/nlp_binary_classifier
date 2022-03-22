"""This module contains all the functions."""

import re
import string
from collections import Counter
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk.corpus import stopwords

from matplotlib.font_manager import FontProperties
font0 = FontProperties(family = 'serif', variant = 'small-caps', size = 26)


def compile_and_fit(model, batch_size, epochs, optimizer, x_train, y_train, x_test, y_test):
    """This function performs the compilation and fit of the model"""

    model.compile(optimizer = optimizer,
                  loss = tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.losses.BinaryCrossentropy(name='loss'),
                           'accuracy',
                           tf.keras.metrics.Precision(name = 'precision'),
                           tf.keras.metrics.Recall(name='recall'),
                           tf.keras.metrics.TruePositives(name='tp'),
                           tf.keras.metrics.TrueNegatives(name='tn'),
                           tf.keras.metrics.FalsePositives(name='fp'),
                           tf.keras.metrics.FalseNegatives(name='fn')
                          ])
    model.summary()

    # early stopping
    e_s = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                           mode='min',
                                           patience=100,
                                           restore_best_weights=True)

    history = model.fit(x_train, y_train,
                        epochs = epochs,
                        verbose = True,
                        validation_data = (x_test, y_test),
                        batch_size = batch_size,
                        callbacks = e_s)
    tf.keras.backend.clear_session()

    return history


def match_and_clean(df_path1, df_path2, key1, key2, df_path):
    """This function matches labels to comments, and performs the cleaning"""

    # Store the data into dataframes
    df_comments = pd.read_csv(df_path1)
    df_efftreat = pd.read_csv(df_path2)

    # Join data frames
    df_comments_efftreat = pd.merge(df_comments, df_efftreat, on="id")

    # Clean the comments
    comments_raw = df_comments_efftreat[key1].values
    comments_clean = clean_comments(comments_raw)

    df_comments_efftreat = df_comments_efftreat.assign(comments_clean = comments_clean)

    # Prepare the labels for efftreat
    labels = np.zeros(len(df_comments_efftreat), dtype = int)
    labels[df_comments_efftreat[key2].values != 0.] = 1
    df_comments_efftreat = df_comments_efftreat.assign(labels=labels)

    # remove empty rows
    df_comments_efftreat = df_comments_efftreat[(df_comments_efftreat['comments_clean'] != '')\
                                                .values]

    # Save the labels
    df_comments_efftreat.to_csv(df_path, columns = ['comments_clean', 'labels'], index = False)

def clean_comments(comments):
    """Clean the comments"""

    comments_clean = ["" for i in range(len(comments))]

    print('Cleaning the comments...')
    with tqdm(total=len(comments)) as pbar:
        for i, comment_i in enumerate(comments):

            try:

                # Remove lower case and Unicode characters
                comment_i = comment_i.lower().encode('ascii', 'ignore').decode()

                # Remove stop words
                stop_words = set(stopwords.words('english'))
                comment_i = ' '.join([word_i for word_i in comment_i.split(' ')
                                    if word_i not in stop_words])

                # Remove:
                to_remove = [
                             r"@\S+", #mention
                             r"https*\S+", #link
                             r"#\S+", #hashtag
                             r"\'\w+", #ticks
                             r"[%s]", #punctuation
                             r"\w*\d+\w*", #number
                             r"\s{2,}", #over space
                             r" +" # first/last blank
                             ]

                for to_remove_i in to_remove:
                    if to_remove_i == "[%s]":
                        comment_i = re.sub(to_remove_i % re.escape(string.punctuation), \
                                                   ' ', comment_i)
                    elif to_remove_i == " +":
                        comment_i = comment_i.strip()
                    else:
                        comment_i = re.sub(to_remove_i, ' ', comment_i)

                # remove single letter words (i, e, a)
                comment_i = [word for word in comment_i.split(' ') if len(word) != 1]

                comments_clean[i] = ' '.join(comment_i)

            except AttributeError:

                pass

            except:

                raise ValueError("It looks like the string '{}'"+\
                                 "could not be handled.".format(comment_i))

            pbar.update(1)

    return comments_clean


def perform_pretreat(data_frame):
    """pre-analyse and pre-treat the data."""

    # Is the dataset balanced?
    df_balance = data_frame.groupby(['labels'], as_index=False).count()

    fig, axs = plt.subplots(1, 1, figsize = (9, 7))
    fig.subplots_adjust(hspace = 0.0, wspace=0., left  = 0.16,
                        right = 0.97, bottom = 0.12, top = 0.98)

    axs.bar(['Useless (label = 0)', 'Usefull (label = 1)'],
             df_balance['comments_clean'].values/len(data_frame)*100.,
             width = 0.4, hatch = ['X', None],
             edgecolor = 'k', linewidth = 2, color = ['#e8a787','#87b9e8'], alpha = 0.7)
    axs.set_ylabel('fraction of the total sample', fontproperties = font0)
    axs.set_xlabel('Classes', fontproperties = font0)
    axs.tick_params(axis='both', labelcolor='k', labelsize = 18,
                    width = 1, size = 20, which = 'major', direction = 'inout')
    axs.tick_params(axis='both', width = 1, size = 10, which = 'minor', direction = 'inout')
    fig.savefig('./plots/data_balance_full_sample.pdf')
    plt.close(fig)

    print('The train data are imbalanced at a level of %i '\
          %((df_balance['comments_clean'].values/len(data_frame)*100.)[0]) +
        'and %i percent for class 0 and class 1, respectively.'\
          %((df_balance['comments_clean'].values/len(data_frame)*100.)[1]))

    ###################
    # This dataset appears to be sightly biased toward positive answers.
    # (see "./plots/data_balance_full_sample.pdf")
    # This is something to keep in mind for later, see if anything needs
    # to be done about this.
    ###################


    # Let's plot the distribution of words.
    df_usefull = data_frame.loc[data_frame['labels'] == 1, 'comments_clean'].tolist()
    df_useless = data_frame.loc[data_frame['labels'] == 0, 'comments_clean'].tolist()

    n_word = 30 #Numbers of top words to look at
    top_usefull = pd.DataFrame(Counter(' '.join(df_usefull).split()).most_common(n_word),
                               columns = ['word', 'count'])
    top_useless = pd.DataFrame(Counter(' '.join(df_useless).split()).most_common(n_word),
                               columns = ['word', 'count'])

    fig, ax1 = plt.subplots(1, 1, figsize = (11, 7))
    fig.subplots_adjust(hspace = 0.0, wspace=0., left  = 0.16,
                        right = 0.97, bottom = 0.28, top = 0.98)

    ax1.bar(top_usefull['word'].values,
          top_usefull['count'].values,
          width = 0.5, edgecolor = 'k', linewidth = 1, color = ['#87b9e8'],
          alpha = 0.7, label = 'Usefull comments')
    ax1.legend()
    ax1.set_ylabel('Counter', fontproperties = font0)
    ax1.tick_params(axis='both', labelcolor='k', labelsize = 18,
                    width = 1, size = 20, which = 'major', direction = 'inout')
    ax1.tick_params(axis='both', width = 1, size = 10, which = 'minor',
                    direction = 'inout')
    ax1.tick_params(axis = 'x', labelrotation=90)
    fig.savefig('./plots/word_counter_UsefullClass.pdf')
    plt.close(fig)

    ###################
    # Staff, care, thank, hospital... are the top hit in the "usefull" comments.
    # (see "./plots/word_counter_UsefullClass.pdf")
    # However, it is possible that these are top hit in "useless" comments too.
    # If that's the case, their predictive power as single word (1-gram) is null.
    # Let's check that.
    ###################

    df_word_counter_all = top_usefull.merge(top_useless, left_on = 'word',
                                            right_on = 'word', suffixes=('_usefull', '_useless'))

    fig, ((ax1), (ax2)) = plt.subplots(2, 1, figsize = (9, 7), sharex = True)
    fig.subplots_adjust(hspace = 0.0, wspace=0., left  = 0.16,
                        right = 0.97, bottom = 0.28, top = 0.98)

    ax1.bar(df_word_counter_all['word'].values,
          df_word_counter_all['count_usefull'].values,
          width = 0.5, edgecolor = 'k', linewidth = 1, color = ['#87b9e8'],
          alpha = 0.7, label = 'Usefull comments')
    ax1.bar(df_word_counter_all['word'].values,
          df_word_counter_all['count_useless'].values,
          width = 0.5, edgecolor = 'k', linewidth = 1, color = ['#e8a787'],
          alpha = 0.7, label = 'Useless comments')

    ax1.legend()
    ax1.set_ylabel('Counter', fontproperties = font0)
    ax1.tick_params(axis='both', labelcolor='k', labelsize = 18,
                    width = 1, size = 20, which = 'major', direction = 'inout')
    ax1.tick_params(axis='both', width = 1, size = 10,
                    which = 'minor', direction = 'inout')

    frac_word_usefull = df_word_counter_all['count_usefull'].values/\
                        sum(df_word_counter_all['count_usefull'].values)
    frac_word_useless = df_word_counter_all['count_useless'].values/\
                        sum(df_word_counter_all['count_useless'].values)
    word_power = frac_word_usefull/(frac_word_usefull + frac_word_useless) - 0.5

    o_index = np.where((word_power > -0.1) & (word_power < 0.1))[0]
    ax2.plot(df_word_counter_all['word'].values[o_index], word_power[o_index], 'o',
             mfc = 'None', mew = 1, mec = 'k')
    o_index = np.where((word_power > 0.1))[0]
    ax2.plot(df_word_counter_all['word'].values[o_index], word_power[o_index], 'o',
             mfc = '#87b9e8', mew = 1, mec = 'k')
    o_index = np.where((word_power < -0.1))[0]
    ax2.plot(df_word_counter_all['word'].values[o_index], word_power[o_index], 'o',
             mfc = '#e8a787', mew = 1, mec = 'k')

    ax2.plot(df_word_counter_all['word'].values,
             np.zeros(len(df_word_counter_all)), 'k--', lw = 1)
    ax2.plot(df_word_counter_all['word'].values,
             np.zeros(len(df_word_counter_all))+0.1, 'k--', lw = 1)
    ax2.plot(df_word_counter_all['word'].values,
             np.zeros(len(df_word_counter_all))-0.1, 'k--', lw = 1)
    ax2.set_ylim([-0.3, 0.3])
    ax2.set_ylabel('word positivity (fraction)', fontproperties = font0, fontsize = 16)
    ax2.tick_params(axis='both', labelcolor='k', labelsize = 18,
                    width = 1, size = 20, which = 'major', direction = 'inout')
    ax2.tick_params(axis='both', width = 1, size = 10, which = 'minor', direction = 'inout')
    ax2.tick_params(axis = 'x', labelrotation=70)
    fig.savefig('./plots/word_counter_twoClass.pdf')
    plt.close(fig)

    ###################
    # Many top words appear in both classes and therefore aren't good discriminant
    # for classification (see top panel of "./plots/word_counter_twoClass.pdf").
    # Perhaps try later to remove them and see if classification improves.
    # Or maybe use n-grams.
    # To get the "word positivity" we calculate the fraction that the word appears
    # in the usefull comments. We do the same for the useless comments. We then
    # calculate the fraction of it appearing in the usefull comments divided
    # by the total fraction it appears in the two classes. We also remove 0.5
    # for normalisation. If close to zero, then it appears at similar levels
    # in usefull and useless comments and potentially have no classification power.
    # (see bottom panel of "./plots/word_counter_twoClass.pdf")
    # Example: Staff appears in 20% of the usefull comments, as well as useless
    # comments. Therefore, 0.2/(0.2+0.2) - 0.5 = 0.
    # For the moment we keep these words and see how classification performs. We
    # might go back to this later and try and remove these.
    ###################


    # Let's see how long the sentences need to be to capture most of the comments.
    frac_words = 0.90 # We try to capture 90% of the length of the comments.

    comments = data_frame['comments_clean'].values
    com_length = [len(comment.split(' ')) for comment in comments]
    y_val, x_val = np.histogram(com_length, bins = 100)

    fig, ax1 = plt.subplots(1, 1, figsize = (11, 7))
    fig.subplots_adjust(hspace = 0.0, wspace=0., left  = 0.16,
                        right = 0.97, bottom = 0.12, top = 0.98)

    y_cumsum = y_val.cumsum()/y_val.cumsum().max()
    # To the nearest 10:
    comments_length = int(np.round(x_val[0:-1][y_cumsum >= frac_words][0]/10.)*10)

    d_x = (x_val[1:] - x_val[:-1])/2.
    ax1.bar(x_val[0:-1] + d_x, y_cumsum, width = d_x*2., edgecolor = 'k',
          linewidth = 1, color = ['#87b9e8'], alpha = 0.4)
    ax1.plot([0., x_val[-1]], [frac_words, frac_words], 'k--', lw = 2)
    ax1.plot([comments_length, comments_length], [0., 1.], 'k--', lw = 2)
    ax1.set_ylabel('Cummulative Distribution', fontproperties = font0)
    ax1.set_xlabel('Number of word in the comments', fontproperties = font0)
    ax1.tick_params(axis='both', labelcolor='k', labelsize = 18,
                    width = 1, size = 20, which = 'major', direction = 'inout')
    ax1.tick_params(axis='both', width = 1, size = 10,
                    which = 'minor', direction = 'inout')

    fig.savefig('./plots/comments_length.pdf')
    plt.close(fig)

    ###################
    # We calculate the length that keeps 90% of the comment lengths.
    # We will use thisd to cut longer comments (see "./plots/comments_length.pdf").
    # We can try to improve classification later on by including
    # these longer comments.
    ###################

    print('90 percent of the comments are less than %i words.'%(comments_length))


    # How long the vocabulary needs to be?
    frac_vocab = 0.9 # We capture 90% of the most important
                     # words (as measured from the number of times it appears).
    vocab = Counter(' '.join(data_frame['comments_clean']).split())
    n_hit_unique = np.unique([val for val in vocab.values()])

    y_val, x_val = np.histogram(n_hit_unique, bins = 200)
    d_x = (x_val[1:] - x_val[:-1])/2.

    n_hit_lim = int(np.interp(frac_vocab,
                              y_val[::-1].cumsum()/y_val[::-1].cumsum().max(),
                              (x_val[0:-1] + d_x)[::-1]))

    fig, ax1 = plt.subplots(1, 1, figsize = (11, 7))
    fig.subplots_adjust(hspace = 0.0, wspace=0., left  = 0.16,
                        right = 0.97, bottom = 0.12, top = 0.98)
    ax1.bar(x_val[0:-1] + d_x, y_val/y_val.max(), width = d_x*2., edgecolor = 'k',
          linewidth = 1, color = ['#87b9e8'], alpha = 0.4)
    ax1.plot([n_hit_lim, n_hit_lim], [0, 1],  'k--', lw = 3)
    ax1.set_ylabel('Number of words (fraction)', fontproperties = font0)
    ax1.set_xlabel('number of times a word appears', fontproperties = font0)
    ax1.tick_params(axis='both', labelcolor='k', labelsize = 18,
                    width = 1, size = 20, which = 'major', direction = 'inout')
    ax1.tick_params(axis='both', width = 1, size = 10,
                    which = 'minor', direction = 'inout')
    ax1.set_yscale('log')
    fig.savefig('./plots/vocab_size.pdf')
    plt.close(fig)

    n_hit = np.array([val for val in vocab.values()])
    word = np.array([key for key in vocab])
    word_in_vocab = sorted(word[n_hit > n_hit_lim])

    ###################
    # One word in the comments appear more than 40 000 times.
    # A lot more appear just one time. Surely those appearing once
    # will not be of a great deal for classification.
    # If we select words that hit more than 216 times, we recover 90 % of
    # the "important" word, where important is measured by the number of times a word appear.
    # (see "./plots/vocab_size.pdf").
    # This corresponds to a vocabulary with 1397 unique words.
    ###################

    print('The length of the vocabulary is limited to the first %i words'%(len(word_in_vocab)))
    print('This was decided by ranking words by order of importance '+\
        '(i.e. fraction of appearance), and keeping 90 percent of it.')

    return comments_length, word_in_vocab
