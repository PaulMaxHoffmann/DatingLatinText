import os
import sys
import glob

# rows = []
# os.chdir('LatLib')
# temp = os.listdir()
# # print(temp)


# # initializing string 
# test_str = '.txt'

# res = []
# for sub in temp:
#     flag = 0
#     for ele in sub:
#         if ele in test_str:
#             flag = 1
#     if not flag:
#         res.append(sub)

# # printing result 
# # print("The list after removal : " + str(res))
# # sys.exit()
# for txt in list(glob.glob("*.txt")):

#     # print(txt)
#     rows.append(txt)

# for i in range(len(res)):
#     os.chdir(f'{res[i]}')
#     for txt in list(glob.glob("*.txt")):

#         # print(txt)
#         rows.append(txt)
#     os.chdir('..')


import numpy as np
import pandas as pd
import os

# data= pd.read_csv('LatLibFakeRankings.csv', encoding= 'latin_1')
data= pd.read_csv('LatLibDates-Filtered.csv', encoding= 'latin_1')

# data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1, inplace=True)
data.rename(columns={'V1': 'Text', 'V2': 'Target'}, inplace=True)

# data['Target']=data['Target'].map({'Easy': 0, 'Medium': 1, 'Hard': 2})

data = data.loc[:500]
# print(data)

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

texts = data['Text']
# labels = data['Target'].astype('category')
labels = data['Target']

from keras.utils import to_categorical
labels = to_categorical(labels)

print("number of texts :" , len(texts))
print("number of labels: ", len(labels))


print(texts[0])

# os.chdir('/home/paul/DeepL/Final/POST')
limit = 140
os.chdir('LatLib')
for i in range(len(texts)):
    if "/" in texts[i]:
        s = texts[i]
        s = s.split('/')
        s1 = s[0]
        s2 = s[1]
        os.chdir(s1)
        with open(s2,'r') as f:
            New_texts = f.read().splitlines()
        texts[i] = New_texts[:limit]
        os.chdir('..')
        # print(f"Sub{i}")
        print(texts[i])
    else:  
        with open(texts[i],'r') as f:
            New_texts = f.read().splitlines()
        texts[i] = New_texts[:limit]
        # print(f"YEE{i}")

