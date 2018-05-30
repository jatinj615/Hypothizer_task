# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


df = pd.read_pickle('df_train_api.pk')

#df = df[df.label != 0]
X = df[['groups']]

y = pd.get_dummies(df['label']).values


from keras.preprocessing.text import Tokenizer
max_features = 4000
tokenizer = Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', split=' ', lower=True, )
tokenizer.fit_on_texts(X['groups'].values)

X = tokenizer.texts_to_sequences(X['groups'].values)

# add padding
from keras.preprocessing.sequence import pad_sequences
X = pad_sequences(X, maxlen=40)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)


#import Keras and important libraries and layers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import SpatialDropout1D
from keras.layers import LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# Using RNN
# Initialising classifier
clf = Sequential()
# Adding First Embedded Layer
clf.add(Embedding(max_features, 128, input_length=X.shape[1]))
clf.add(SpatialDropout1D(0.9))

# Adding Lstm Layer
clf.add(LSTM(128, dropout=0.8, recurrent_dropout=0.8))

# Adding fully connected layer
clf.add(Dense(128, activation='relu'))

# Adding output layer
clf.add(Dense(6, activation='softmax'))

# Compiling classifier
clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# importing tensorflow for training on gpu
import tensorflow as tf
with tf.device('/gpu:0'):
    clf.fit(X_train, y_train, batch_size=32, epochs=40, validation_data= (X_test, y_test))

def build_clf():
    clf = Sequential()
    clf.add(Embedding(max_features, 128, input_length=X.shape[1]))
    clf.add(SpatialDropout1D(0.6))
    clf.add(LSTM(128, dropout=0.6, recurrent_dropout=0.5))
    clf.add(Dense(128, activation='relu'))
    clf.add(Dense(6, activation='softmax'))
    clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return clf

clf = KerasClassifier(build_fn= build_clf, batch_size = 32, epochs = 30)
scores = cross_val_score(estimator= clf, X = X_train, y = y_train, cv = 10, n_jobs = -1)

mean = np.mean(scores)
variance = np.std(scores)


y_pred = clf.predict(X_test)
y_pred = ( y_pred > 0.5 )
y_pred_0 = y_pred[:, 0]
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
acc = accuracy_score(y_test[:, 0], y_pred[:, 0])
cf_report = classification_report(y_test[:, 0], y_pred[:, 0])
cm = confusion_matrix(y_test[:, 0], y_pred[:, 0])