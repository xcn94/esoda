import os
import string
import heapq
import numpy as np
from pprint import pprint

path = '/root/xcn/file4lab/esoda/datasets/'
root_folders = os.listdir(path)
test_folder = 'conf_chi'
test_file = 'conf_chi_Densmore12.txt'

stopwords_path = '/root/xcn/file4lab/esoda/datasets/stopwords.txt'
stopwords = []
with open(stopwords_path) as f:
    for word in iter(f):
        stopwords.append(word.strip())

sentences = []

# for folder in root_folders:
#
#     files = os.listdir(path + folder)
#     for file in files:
#         print(file)
#         f = open(path + folder + '/' + file)
#         iter_f = iter(f)
#         for line in iter_f:
#             sentences.append(line)
#
# print(len(sentences))
input_text = ''
translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
with open(path + test_folder + '/' + test_file) as f:
    for line in iter(f):
        line = line.translate(translator)
        input_text += ' ' + line.strip()
text_list = input_text.lower().split()
text_list = [word for word in text_list if word not in stopwords]

text_set = set(text_list)
print(len(text_list))
print(len(text_set))

textset_length = len(text_set)

# for word in text_set:
#     if word not in stopwords:
#         print(word)

word2id = {}
id2word = {}
word_totalWeights = {}
word_graph = np.zeros((textset_length, textset_length), dtype=int)

for i, word in enumerate(text_set):
    word2id[word] = i
    id2word[i] = word

ngram = 3
for i in range(len(text_list)):
    inode = word2id[text_list[i]]
    for j in range(1-ngram, ngram-1):
        if j!=0 and i+j>=0 and i+j<len(text_list):
            onode = word2id[text_list[i+j]]
            word_graph[inode][onode] += 1
            word_graph[onode][inode] += 1

for word in text_set:
    idx = word2id[word]
    weight = np.sum(word_graph[idx])
    word_totalWeights[word] = weight

# pprint(word_totalWeights)

WS = np.ones(textset_length, dtype=float)
d = 0.85

# while True:
for _ in range(20):
    ws_loss = 0
    for i in range(len(WS)):
        WS_old = WS[i]
        updateWS = 0.0
        for j in range(len(WS)):
            if j != i and word_graph[i][j]:
                updateWS += word_graph[i][j] * WS[j] / word_totalWeights[id2word[j]]
        WS[i] = (1-d) + d * updateWS

        # if (WS[i] - WS_old) / WS_old < 1e-5:
        #     break
max_list = map(list(WS).index, heapq.nlargest(20, list(WS)))
max_words = [id2word[i] for i in max_list]
print(max_words)
