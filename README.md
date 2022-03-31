**Binary classification using NLP**

This is a binary classifier that uses NLP to classify comments reviewing healthcare practices. The classifier determines whether a comment is useful or not to measure the effectiveness of the treatment administered.

This is trained on data that I am not allowed to share, but which have been pre-labelled as "useful" or "not useful". The aim is to classify a new unseen comment as "useful" or "not useful".

I will constantly build upon the latest version by optimising the network metrics. There is always plenty of room to improve, by starting with using different models and word embeddings, but also by using a tailored data preparation. In our case we want to maximise precision, which will reduce the recall. This is because it is better to obtain a clean sample of useful comments (minimising the false positives), at the expense of losing some useful comments.

---------------
This is V2.0.0:

**data preparation**:\
Text normalisation + defining a vocabulary.

**word embedding**:\
Binary bag of words: convert the comments into sparse matrices of zeros and ones. Zero is put in place if the word is not in the vocabulary, and one is placed if the word is in the vocabulary.

Example:\
vocabulary = ['red', 'car', 'pigeon']\
comment = ['jon has a red car']\
embedding = [0,0,0,1,1]


**Model**:\
basic 16/1 Dense layers. (see ./histories/model_V2p0p0.txt)

**Performances on the test data for 500 epochs:**
* Reach over-fitting very quickly (< 100 epochs), but had an early stop.
* Loss: 0.35
* Accuracy: 86%
* Precision: 89%
* Recall: 91%

**Changes/Comments**:\
From V1.0.0 only the word embedding has changed. The differences in performances can be seen on the figure "histories/history_V1p0p0_vs_V2p0p0.pdf". Both the precision and accuracy have marginally increased, which is promising. Future work will focus on trying other word embeddings, as well as adding some penalising factors in the NN to avoid the fast over-fitting.

---------------
This is V1.0.0:

**data preparation**:\
text normalisation + defining a vocabulary.

**word embedding**:\
Here I used a simple word embedding where each of the words in the vocabulary is one-hot encoded. Therefore, one can encode a full comment by replacing each of the words by the one-hot encoded word. For simplicity (and speed), this first version further average over the comments (i.e. all of the words in a comment are collapsed into a single vector).

Example:\
vocabulary = ['red', 'car', 'pigeon']\
comment = ['jon has a red car']\
matrix = [0,0,0] (jon)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[0,0,0] (has)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[0,0,0] (a)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[1,0,0] (red)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[0,1,0] (car)\
embedding = [0.2, 0.2, 0.] (averaged over the sentence)

**Model**:\
Basic 16/1 Dense layers. (see ./histories/model_V1p0p0.txt)

**Performances on the test data for 500 epochs:**
* No apparent over-fitting, but stopped learning.
* Loss: 0.37
* Accuracy: 84%
* Precision: 87%
* Recall: 91%

---------------
