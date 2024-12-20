import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Input
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
import os
import string
from random import randint
from pickle import load

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def clean_doc(doc):
    doc = doc.replace('--', ' ')
    tokens = doc.split()
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    return tokens

def save_doc(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

in_filename = 'ThreeMusketeers.txt'
doc = load_doc(in_filename)
print(doc[:200])

tokens = clean_doc(doc)
print(tokens[:200])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))

length = 50 + 1
sequences = list()
for i in range(length, len(tokens)):
    seq = tokens[i-length:i]
    line = ' '.join(seq)
    sequences.append(line)

print('Total Sequences: %d' % len(sequences))

out_filename = 'ThreeMusketeers.txt'
save_doc(sequences, out_filename)

in_filename = 'ThreeMusketeers.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
vocab_size = len(tokenizer.word_index) + 1

sequences = np.array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

X = np.reshape(X, (X.shape[0], X.shape[1], 1))
inputs = Input(shape = (X.shape[1], X.shape[2]))
x = LSTM(256)(inputs)
x = Dropout(0.2)(x)
outputs = Dense(vocab_size, activation = 'softmax')(x)
model = Model(inputs, outputs)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(X, y, epochs=20, batch_size=128)
model.save('/Assignment4model') # save model to file