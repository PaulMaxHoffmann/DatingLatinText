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


data= pd.read_csv('LatLibDates-Filtered.csv', encoding= 'latin_1')

data.rename(columns={'V1': 'Text', 'V2': 'Target'}, inplace=True)


texts = data['Text']

labels = data['Target']

print("number of texts :" , len(texts))

print(texts[0])
Names = []
for i in range(len(texts)):
    Names.append(texts[i])

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
        texts[i] = New_texts[100:]
        os.chdir('..')
        # print(f"Sub{i}")
    else:  
        with open(texts[i],'r') as f:
            New_texts = f.read()
            texts[i] = New_texts[100:]
        # print(f"YEE{i}")
UWU = []
for i in range(len(texts)):
    texts[i] = RegEX(texts[i]) 

for i in range(len(Names)):
    Names[i] = Names[i].replace('/','') 
os.chdir('/home/paul/DeepL/Final2/Final-LatLib')
for i in range(len(texts)):
    if len(texts[i]) >= 10000:
        # print(Names[i])
        chunk_lim = len(texts[i])//10000
        # print(chunk_lim)
        chunk_text = [''.join(item) for item in zip(*[iter(texts[i])]*10000)]
        # print(chunk_text[0])
        for j in range(chunk_lim):
            with open(f"{Names[i]}_{j}.txt", "w") as text_file:
                print(chunk_text[j], file=text_file)
            UWU.append(f"{Names[i]}_{j}.txt")
    else:
        with open(f"{Names[i]}.txt", "w") as text_file:
            print(texts[i], file=text_file)
        UWU.append(f"{Names[i]}.txt")
    


# import csv
with open(os.path.join('/home/paul/DeepL/Final2','Final_Rank.csv'),'w') as f:
    for line in UWU:
        f.write(line)
        f.write('\n')

