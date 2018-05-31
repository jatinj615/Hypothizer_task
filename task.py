# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


df = pd.read_pickle('df_train_api.pk')

df0 = df.query('label==0').sample(frac = 0.1)
df = df[df.label != 0]
df = pd.concat([df, df0], ignore_index=True)
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

# Using ANN
# Initialising classifier
clf = Sequential()
# Adding Input layer and first hidden layer
clf.add(Dense(output_dim = 25, kernel_initializer='uniform', activation='relu', input_dim = 40))
clf.add(Dropout(rate=0.1))
# Adding fully connected layer
clf.add(Dense(output_dim = 25, activation='relu', kernel_initializer='uniform'))
clf.add(Dropout(rate=0.1))
# Adding output layer
clf.add(Dense(6,kernel_initializer='uniform', activation='softmax'))

# Compiling classifier
clf.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# importing tensorflow for training on gpu
import tensorflow as tf
with tf.device('/gpu:0'):
    clf.fit(X_train, y_train, batch_size=10, epochs=1000, validation_data= (X_test, y_test))



y_pred = clf.predict(X_test)
y_pred = ( y_pred > 0.5 )
y_pred_0 = y_pred[:, 0]
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
acc = accuracy_score(y_test[:, 0], y_pred[:, 0])
cf_report = classification_report(y_test[:, 0], y_pred[:, 0])
cm = confusion_matrix(y_test[:, 0], y_pred[:, 0])