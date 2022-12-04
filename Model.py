import numpy as np
import pandas as pd
import os
import re

def RegEX(string):
    file_1_str = re.sub(r"[([{})]]", "", string) #removing brackets
    file_1_str = re.sub("\d+", "", file_1_str) #removing numbers
    file_1_str = re.sub(r"\s+", " ", file_1_str)#removing tabs
    file_1_str = re.sub(r'[^\w\s]', '', file_1_str)#remove punc
    return re.sub('\u0304', '', file_1_str)

# data= pd.read_csv('LatLibFakeRankings.csv', encoding= 'latin_1')
data= pd.read_csv('LatLibDates-Filtered.csv', encoding= 'latin_1')

# data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1, inplace=True)
data.rename(columns={'V1': 'Text', 'V2': 'Target'}, inplace=True)

# data['Target']=data['Target'].map({'Easy': 0, 'Medium': 1, 'Hard': 2})

data = data.loc[:200]
# print(data)

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

texts = data['Text']
# labels = data['Target'].astype('category')
labels = data['Target']

for i in range(len(labels)):
    if labels[i] <= 5.99:
        labels[i] = 1
    elif 6 <= labels[i] <= 8.99:
        labels[i] = 2
    elif 9 <= labels[i] <= 10.99:
        labels[i] = 3
    elif 11 <= labels[i] <= 14.99:
        labels[i] = 4
    elif 15 <= labels[i] <= 18.99:
        labels[i] = 5
    elif 19 <= labels[i] <= 27.99:
        labels[i] = 6

# print(labels)
# import sys
# sys.exit()

from keras.utils import to_categorical
labels = to_categorical(labels)

print("number of texts :" , len(texts))
print("number of labels: ", len(labels))


print(texts[0])

# os.chdir('/home/paul/DeepL/Final/POST')
limit = 2000
os.chdir('LatLib')
for i in range(len(texts)):
    if "/" in texts[i]:
        s = texts[i]
        s = s.split('/')
        s1 = s[0]
        s2 = s[1]
        os.chdir(s1)
        with open(s2,'r') as f:
            New_texts = f.read()
        # texts[i] = New_texts
        texts[i] = New_texts[:limit]
        os.chdir('..')
        # print(f"Sub{i}")
    else:  
        with open(texts[i],'r') as f:
            New_texts = f.read()
        # texts[i] = New_texts
        texts[i] = New_texts[:limit]
        # print(f"YEE{i}")

# print(texts[0])
for i in range(len(texts)):
    texts[i] = RegEX(texts[i]) 
# print("HOWDY")  
# print(texts[0])   
# import sys
# sys.exit()

from sklearn import preprocessing
import numpy as np


tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

vocab_size = len(tokenizer.word_index) + 1

sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print("Found {0} unique words: ".format(len(word_index)))

seqs = pad_sequences(sequences)

normalized_arr = preprocessing.normalize(seqs)
seqs = normalized_arr
# print(seqs)

print("data shape: ", seqs.shape)

#10. Read the embeddings in the pretrained model 


# import json

# os.chdir('..')
# with open('output.txt', 'w') as filehandle:
#     json.dump(seqs.tolist(), filehandle)

# import sys
# sys.exit()

# path_to_glove_file = "glove.6B.100d.txt"
path_to_word2vec_file = '/home/paul/DeepL/Final/Word2Vec.vec'

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

    p_pred = model.predict(x_test)
    # p_pred = p_pred.flatten()
    # print(p_pred.round(2))
    # [1. 0.01 0.91 0.87 0.06 0.95 0.24 0.58 0.78 ...

    # extract the predicted class labels
    # y_pred = np.where(p_pred > 0.5)
    y_pred = (p_pred > 0.5)
    # print(y_pred)
    # [1 0 1 1 0 1 0 1 1 0 0 0 0 1 1 0 1 0 0 0 0 ...

    print("Confusion Matrix")
    # print(confusion_matrix(y_test, y_pred))
    print(confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
    print(classification_report(y_test, y_pred))

# define the model LSTM

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
x = layers.Bidirectional(layers.LSTM(120, return_sequences=True))(embedded_sequences)
x = layers.Bidirectional(layers.LSTM(120))(x)
# before = layers.Dense(20, activation="relu")(x)
preds = layers.Dense(7, activation="softmax")(x)
model = Model(int_sequences_input, preds)


#define model
# model = Sequential()

# embedding_layer = Embedding(vocab_size, EMBEDDING_SIZE,
#                             embeddings_initializer= Constant(embedding_matrix), 
#                             trainable=False,input_length = 314
# )
# model.add(embedding_layer)
# model.add(LSTM(128, activation ='relu'))
# # model.add(LSTM(32, input_shape=(None,128), activation ='relu'))
# model.add(Dense(128, activation='relu'))
# # model.add(Dropout(0.2))
# model.add(Dense(32, activation='relu'))
# # model.add(Dense(32, activation='relu'))
# model.add(Flatten())   #add Flatten layer 
# model.add(Dense(7, activation='softmax'))


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

history = model.fit(X_train, Y_train, epochs = 5, batch_size = 128)

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