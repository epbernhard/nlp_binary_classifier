import matplotlib.pyplot as plt
import pickle
import numpy as np

history = pickle.load(open('./histories/history_one_300_64_one_hot', "rb"))

epoch = np.arange(0, len(history['loss']))

fig = plt.figure(figsize=(20, 10))

plt.subplot(2,2,1)
plt.plot(epoch, history['loss'], color = 'k', label='Train: history V1.0, batch: 64, l_r = 0.01', alpha = 0.6)
plt.plot(epoch, history['val_loss'], '--', color = 'k', label='Test: history V1.0, batch: 64, l_r = 0.01', alpha = 0.6)
plt.xlabel('Epoch', fontsize = 20)
plt.ylabel('Loss', fontsize = 20)
plt.legend(fontsize = 15)

plt.subplot(2,2,2)
plt.plot(epoch, history['accuracy'], color = 'k', alpha = 0.6)
plt.plot(epoch, history['val_accuracy'], '--', color = 'k', alpha = 0.6)
plt.xlabel('Epoch', fontsize = 20)
plt.ylabel('Accuracy', fontsize = 20)


plt.subplot(2,2,3)
plt.plot(epoch, history['precision'], color = 'k', alpha = 0.6)
plt.plot(epoch, history['val_precision'], '--', color = 'k', alpha = 0.6)
plt.xlabel('Epoch', fontsize = 20)
plt.ylabel('Precision', fontsize = 20)

plt.subplot(2,2,4)
plt.plot(epoch, history['recall'], color = 'k', alpha = 0.6)
plt.plot(epoch, history['val_recall'], '--', color = 'k', alpha = 0.6)
plt.xlabel('Epoch', fontsize = 20)
plt.ylabel('recall', fontsize = 20)

fig.savefig('./histories/history_V1.0.0.pdf')
