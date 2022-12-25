import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns

data= pd.read_csv('Final_Rank_Balanced.csv', encoding= 'latin_1')


data.rename(columns={'V1': 'Text', 'V2': 'Target'}, inplace=True)

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

# data = data.sample(frac=1,random_state=1).reset_index()
# print(data.head())
# print(shuffled.head())


texts = data['Text']
labels = data['Target']

from tensorflow.keras.utils import to_categorical
labels = to_categorical(labels)

print("number of texts :" , len(texts))
print("number of labels: ", len(labels))

os.chdir('Final-LatLib2')
for i in range(len(texts)):
    with open(texts[i],'r') as f:
        New_texts = f.read()
    texts[i] = New_texts[:500]

from sklearn import preprocessing
import numpy as np

print(texts[1])


tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

vocab_size = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print("Found {0} unique words: ".format(len(word_index)))

seqs = pad_sequences(sequences)

print("data shape: ", seqs.shape)
print(seqs[1])
# import sys
# sys.exit()
os.chdir('..')
path_to_word2vec_file = 'Word2Vec.vec'

embeddings_index = {}
with open(path_to_word2vec_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))


embedding_matrix = np.zeros((vocab_size, 300))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print(embedding_matrix.shape)

#Splitting the data
from sklearn.model_selection import train_test_split
X_train, x_test, Y_train, y_test = train_test_split(seqs, labels, test_size=0.3, shuffle=True)

#Using Neural Networks
from tensorflow import keras
from tensorflow.keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, LSTM, SimpleRNN, GRU

from keras.models import Sequential
from keras.layers import Dense, Flatten
# from keras.layers.embeddings import Embedding

from keras import layers, Input, Model


from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report


def gen_conf_matrix(model, x_test, y_test):

    predictions = model.predict(x_test, steps=len(x_test), verbose=0)
    y_pred = np.argmax(predictions, axis=-1)

    y_true=np.argmax(y_test, axis=-1)

    cm = confusion_matrix(y_true, y_pred)

    ## Get Class Labels

    class_names = [1,2,3,4,5,6,7,8]

    # Plot confusion matrix in a beautiful manner
    fig = plt.figure(figsize=(6, 6))
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Predicted', fontsize=20)
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=90)
    ax.xaxis.set_ticklabels(class_names, fontsize = 10)
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True', fontsize=20)
    ax.yaxis.set_ticklabels(class_names, fontsize = 10)
    plt.yticks(rotation=0)

    plt.title('Refined Confusion Matrix', fontsize=20)

    plt.savefig('Final_50epoch_500ch.png')
    plt.show()

EMBEDDING_SIZE = 300


from keras.models import Sequential
from keras.layers import Dense, Flatten
# from keras.layers.embeddings import Embedding

from keras.layers import Embedding
from keras.initializers import Constant


embedding_layer = Embedding(vocab_size, EMBEDDING_SIZE,
                            embeddings_initializer= Constant(embedding_matrix), 
                            trainable=False,
)

int_sequences_input = Input(shape=(None,), dtype="int64")
embedded_sequences = embedding_layer(int_sequences_input)
x = layers.Bidirectional(layers.LSTM(1024, return_sequences=True))(embedded_sequences)
x = layers.Bidirectional(layers.LSTM(1024))(x)
# before = layers.Dense(20, activation="relu")(x)
preds = layers.Dense(9, activation="softmax")(x)
model = Model(int_sequences_input, preds)





# summarize the model
model.summary()
# model.compile(loss = 'binary_crossentropy', optimizer ='adam',metrics = ["accuracy",f1_m,precision_m, recall_m])
#optimizer = 'SGD'
model.compile(loss = 'categorical_crossentropy', optimizer ='adam',metrics = ["accuracy",f1_m,precision_m, recall_m])

#9. Train and save the best model
# from keras.callbacks import ModelCheckpoint
# filepath = "LSTM_EM_model.h1"
# checkpoint = ModelCheckpoint(filepath, monitor = "loss", mode = "min", verbose =1, save_best_only = True)

# history = model.fit(X_train, Y_train, epochs = 10, batch_size = 100, callbacks = [checkpoint])

history = model.fit(X_train, Y_train, epochs = 50, batch_size = 32)

#Full
print("Score of the total test data")
score = model.evaluate(x_test, y_test, verbose = 0)
# loss, accuracy, f1_score, precision, recall
print("Test loss: %.4f" % score[0])
print("Test accuracy: %.2f" % (score[1] * 100.0))
print("Test f1_score: %.2f" % (score[2]))
print("Test precision: %.2f" % (score[3]))
print("Test recall: %.2f" % (score[4]))
gen_conf_matrix(model, x_test, y_test)