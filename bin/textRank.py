class textRank():

    def __init__(self, text, stopwords, ngram=3, d=0.85):
        '''
            input
             text: text string
             ngram: countering on N-1 words both on front and back sides
        '''
        self.input_text = text
        self.stopwords = stopwords
        self.ngram = ngram
        self.word2id = {}
        self.id2word = {}
        self.word_totalWeights = {}
        self.text_list = []
        self.text_set = set()
        self.textset_length = 0
        self.d = d

    def getText(self):
        self.text_list = self.input_text.lower().split()
        self.text_list = [word for word in self.text_list if word not in self.stopwords]
        self.text_set = set(self.text_list)
        self.textset_length = len(self.text_set)

    def run(self):
        import numpy as np
        import heapq
        self.getText()
        word_graph = np.zeros((self.textset_length, self.textset_length), dtype=int)
        for i, word in enumerate(self.text_set):
            self.word2id[word] = i
            self.id2word[i] = word
        for i in range(len(self.text_list)):
            inode = self.word2id[self.text_list[i]]
            for j in range(1 - self.ngram, self.ngram - 1):
                if j != 0 and i + j >= 0 and i + j < len(self.text_list):
                    onode = self.word2id[self.text_list[i + j]]
                    word_graph[inode][onode] += 1
                    word_graph[onode][inode] += 1

        for word in self.text_set:
            idx = self.word2id[word]
            weight = np.sum(word_graph[idx])
            self.word_totalWeights[word] = weight

        # pprint(word_totalWeights)

        WS = np.ones(self.textset_length, dtype=float)
        # while True:
        for _ in range(40):
            for i in range(len(WS)):
                WS_old = WS[i]
                updateWS = 0.0
                for j in range(len(WS)):
                    if j != i and word_graph[i][j]:
                        updateWS += word_graph[i][j] * WS[j] / self.word_totalWeights[self.id2word[j]]
                WS[i] = (1 - self.d) + self.d * updateWS

                # if (WS[i] - WS_old) / WS_old < 1e-5:
                #     break
        max_list = map(list(WS).index, heapq.nlargest(20, list(WS)))
        max_words = [self.id2word[i] for i in max_list]
        print(max_words)
