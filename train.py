import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from keras.layers import Activation, Dense
from keras.models import Sequential
from web_processor import WebProcessor

filename = 'data/data.txt'
query = 'artificial intelligence'.split()
with open(filename) as f:
    lines = f.read().splitlines()
data = list(
    map(lambda x: (float(x.split('\t')[0]), eval(x.split('\t')[1])), lines))

wp = WebProcessor(query=query)
label = np.empty(len(data))
features = np.empty([len(data), len(wp.tags)])

for idx, (score, tag_text) in enumerate(data):
    label[idx] = score
    features[idx] = wp.get_tfidf(tag_text)

x_train, x_test, y_train, y_test = train_test_split(
    features, label, shuffle=True)
print('Train size:', len(x_train))
print('Test size:', len(x_test))
print('features.shape', features.shape)
print('label.shape', label.shape)

epochs = 5
batch_size = 32

model = Sequential()

model.add(Dense(12, input_shape=(features.shape[1], ), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, ))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

history = model.fit(
    x_train, y_train, epochs=epochs, validation_split=0.2, verbose=1)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# Calculates and prints r2 score of training and testing data
print("The R2 score on the Train set is:\t{:0.3f}".format(
    r2_score(y_train, y_train_pred)))
print("The R2 score on the Test set is:\t{:0.3f}".format(
    r2_score(y_test, y_test_pred)))
