This is a binary classifier that uses NLP to classify comments reviewing healthcare practices. The classifier determines whether a comment is usefull or not to measure the effectiveness of the treatment administred.

This is trained on data that I am not allowed to share, but which has been pre-labbeled as usefull or not usefull.

This is V1.0.0:

data preparation: text normalisation + defining a vocabulary.

word embedding: here I used a simple word embedding where each of the words in the vocabulary is one-hot encoded. Therefore, one can encode a full comment by replacing each of the words by the one-hot encoded word. For simplicity (and speed), this first version further average over the comments (i.e. all of the words in a comment are collapsed into a single vector).

Model: basic 32/16/1 Dense layers.

Performances on the test data for 300 epochs:
	- No apparent over-fitting, but stopped learning.
	- Loss: 0.37
	- Accuracy: 84%
	- Precision: 87%
	- Recall: 91%

Building upon this, I will update this model to optimise these metrics. There is plenty of room to improve, by starting with using different models and word embeddings, but also by using a tailored data preparation. In our case we want to maximise precision, which will reduce the recall. This is because it is better to obtain a clean sample of usefull comments (minimising the false positives), at the expense of losing some usefull comments in counterpart.