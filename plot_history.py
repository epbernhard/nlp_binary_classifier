import matplotlib.pyplot as plt
import pickle
import numpy as np

v1 = 'V1p0p0'
label_1 = 'V1.0.0'
history = pickle.load(open('./histories/history_'+v1, "rb"))

v2 = 'V2p0p0'
label_2 = 'V2.0.0'
history_comp = pickle.load(open('./histories/history_'+v2, "rb"))

epoch = np.arange(0, len(history['loss']))
epoch_comp = np.arange(0, len(history_comp['loss']))

fig = plt.figure(figsize=(20, 10))

plt.subplot(2,2,1)
plt.plot(epoch, history['loss'], color = 'k', label= label_1 + '(Train)', alpha = 0.6)
plt.plot(epoch, history['val_loss'], '--', color = 'k', label= label_1  + '(Test)', alpha = 0.6)
plt.plot(epoch_comp, history_comp['loss'], color = '#60a6db', label= label_2 + '(Train)', alpha = 0.6)
plt.plot(epoch_comp, history_comp['val_loss'], '--', color = '#60a6db', label= label_2  + '(Test)', alpha = 0.6)
plt.xlabel('Epoch', fontsize = 20)
plt.ylabel('Loss', fontsize = 20)
plt.legend(fontsize = 15, ncol = 2)

plt.subplot(2,2,2)
plt.plot(epoch, history['accuracy'], color = 'k', alpha = 0.6)
plt.plot(epoch, history['val_accuracy'], '--', color = 'k', alpha = 0.6)
plt.plot(epoch_comp, history_comp['accuracy'], color = '#60a6db', alpha = 0.6)
plt.plot(epoch_comp, history_comp['val_accuracy'], '--', color = '#60a6db', alpha = 0.6)
plt.xlabel('Epoch', fontsize = 20)
plt.ylabel('Accuracy', fontsize = 20)


plt.subplot(2,2,3)
plt.plot(epoch, history['precision'], color = 'k', alpha = 0.6)
plt.plot(epoch, history['val_precision'], '--', color = 'k', alpha = 0.6)
plt.plot(epoch_comp, history_comp['precision'], color = '#60a6db', alpha = 0.6)
plt.plot(epoch_comp, history_comp['val_precision'], '--', color = '#60a6db', alpha = 0.6)
plt.xlabel('Epoch', fontsize = 20)
plt.ylabel('Precision', fontsize = 20)

plt.subplot(2,2,4)
plt.plot(epoch, history['recall'], color = 'k', alpha = 0.6)
plt.plot(epoch, history['val_recall'], '--', color = 'k', alpha = 0.6)
plt.plot(epoch_comp, history_comp['recall'], color = '#60a6db', alpha = 0.6)
plt.plot(epoch_comp, history_comp['val_recall'], '--', color = '#60a6db', alpha = 0.6)
plt.xlabel('Epoch', fontsize = 20)
plt.ylabel('recall', fontsize = 20)

fig.savefig('./histories/history_'+v1+'_vs_'+v2+'.pdf')
