#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import re
import hashlib
import json
import time

import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split

import keras
from keras import Input
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import Dense, Embedding, Conv1D, Conv2D, Multiply, GlobalMaxPooling1D, Dropout, Activation, RNN, LSTM, Bidirectional
from keras.layers import UpSampling2D, Flatten, merge, MaxPooling2D, MaxPooling1D, UpSampling1D, AveragePooling1D, GlobalMaxPooling2D
from keras.models import load_model, Model
from keras.layers import merge, Dropout, BatchNormalization, Maximum, Add, Lambda, Concatenate
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.engine.topology import Layer
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


# In[ ]:


# specify which GPU will be used
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


# In[ ]:


# load labels
labels = pd.read_csv("../dataset/label.csv")


# In[ ]:


X_May = []
X_Apr = []
for row in labels.iterrows():
    file_name = row[1]['file_name']
    label = row[1]['is_malicious']
    if file_name.startswith("201704"):
        X_Apr.append({"file_name": file_name, "label": label})
    else:
        X_May.append({"file_name": file_name, "label": label})
X_Apr = pd.DataFrame(X_Apr)
X_May = pd.DataFrame(X_May)


# In[ ]:


max_length = 1000

class ClassifyGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, datasets, labels, batch_size=32, dim=max_length, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.datasets = datasets
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, self.dim, 102), dtype=float)
        X2 = np.zeros((self.batch_size, self.dim, 12), dtype=float)
        y = np.zeros(self.batch_size, dtype=float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            base_path = "../dataset/dataset/{0}.npy"
            item = self.datasets.iloc[ID]
            self_name = item['file_name']
            tmp = np.load(base_path.format(self_name))
            tmp = np.clip(tmp, -100, 100)
            if tmp.shape[0] > self.dim:
                X[i] = tmp[:self.dim, :]
                X2[i] = tmp[:self.dim, :12]
            else:
                X[i, :tmp.shape[0], :] = tmp[:, :]
                X2[i, :tmp.shape[0], :] = tmp[:, :12]
            y[i] = self.labels[ID]
            
        return X, y


# In[ ]:


class Model():
    def __init__(self):
        self.start_time = time.time()

    def get_model(self, model_path = None):  
        if model_path is None:
            params_input = Input(shape=(max_length, 102))
            
            x = BatchNormalization()(params_input)
            
            x_0 = Conv1D(128, 2, strides=1, padding='same')(x)
            x_1 = Conv1D(128, 2, strides=1, activation="sigmoid", padding='same')(x)
            gated_0 = Multiply()([x_0, x_1])
            
            x_0 = Conv1D(128, 3, strides=1, padding='same')(x)
            x_1 = Conv1D(128, 3, strides=1, activation="sigmoid", padding='same')(x)
            gated_1 = Multiply()([x_0, x_1])
            
            x  = Concatenate()([gated_0, gated_1])
            x = BatchNormalization()(x)
            
            x = Bidirectional(LSTM(100, return_sequences=True))(x)
            
            x = GlobalMaxPooling1D()(x)

            x = Dense(64)(x)
            x = Activation('relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(1)(x)

            net_output = Activation('sigmoid')(x)

            model = keras.models.Model(inputs=[params_input], outputs=net_output)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        else:
            model = load_model(model_path)
            
        model.summary()
        return model

    def train(self, max_epoch, batch_size, x_train, y_train, x_val, y_val, x_test, y_test):
        model = self.get_model()
        class_name = self.__class__.__name__

        print('Length of the train: ', len(x_train))
        print('Length of the validation: ', len(x_val))
        print('Length of the test: ', len(x_test))
        
        training_generator = ClassifyGenerator(range(len(x_train)), x_train, y_train, batch_size)
        validation_generator = ClassifyGenerator(range(len(x_val)), x_val, y_val, batch_size, shuffle=False)
        
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        callbacks_list = [es]
        
        #If the program is running on Windows OS, you can remove "use_multiprocessing=True," and "workers=6,".
        model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            use_multiprocessing=True,
                            epochs=max_epoch,
                            workers=6,
                            callbacks=callbacks_list
                           )
        return model


# In[ ]:

from sklearn.model_selection import KFold,StratifiedKFold

n_fold = 4
folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=0)
y = X_Apr.label.ravel()
X = X_Apr.drop(columns=['label'])
y_test = X_May.label.ravel()
x_test = X_May.drop(columns=['label'])

for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
    x_train, x_val = X.iloc[train_index].reset_index(drop=True), X.iloc[valid_index].reset_index(drop=True)
    y_train, y_val = y[train_index], y[valid_index]
    cnn_lstm_model = Model().train(25, 64, x_train, y_train, x_val, y_val, x_test, y_test)
    cnn_lstm_model.save("cnn_lstm_model_" + str(fold_n) + ".h5")



# In[ ]:

import seaborn as sns
from sklearn.metrics import accuracy_score

sns.set_style('white')

def plot_recall(y_true, y_pred):
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    fpr, tpr, threshold = roc_curve(y_true[:len(y_pred)], y_pred)
    roc_auc = auc(fpr, tpr)
    
    tmp_df = pd.DataFrame({'fpr':fpr,'tpr':tpr}).groupby('fpr').max()
    tpr = tmp_df['tpr'].ravel()
    fpr = tmp_df.index.ravel()

    idx = find_nearest(fpr, 0.001)
    print("fpr", fpr[idx])

    plt.figure()
    lw = 2
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='#fb8072',
             lw=lw, label='AUC=%0.5f' % roc_auc, linestyle='-')

    plt.xlim([0.0, 0.01])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('Recall')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.vlines(fpr[idx], 0, tpr[idx], colors = '#b3de69', linestyles = "dashed")
    plt.hlines(tpr[idx], 0, fpr[idx], colors = '#b3de69', linestyles = "dashed")
    plt.annotate(r'$recall={:.5f}$'.format(tpr[idx]), xy=(0.0004, tpr[idx]), xycoords='data', xytext=(-10, +20),
             textcoords='offset points', fontsize=12)
    plt.annotate(r'$fpr=%.4f$' % fpr[idx], xy=(0.0004, 0.4), xycoords='data', xytext=(+30, -30),
             textcoords='offset points', fontsize=12)
    print("tpr", tpr[idx])
    print("auc", roc_auc)

    plt.show()

# In[ ]:

def predict(model_name, data, label):
    model = load_model(model_name)
    validation_generator = ClassifyGenerator(range(len(data)), data, label, 10, shuffle=False)
    y_pred = model.predict_generator(generator=validation_generator, max_queue_size=10, verbose=1)
    return y_pred

for fold_n in range(n_fold):
    model_name = "cnn_lstm_model_" + str(fold_n) + ".h5"
    y_pred = predict(model_name, x_test, y_test)
    print("acc", accuracy_score((y_pred>0.5).astype('int'), y_test[:len(y_pred)]))
    plot_recall(y_test, y_pred)
