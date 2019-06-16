from pprint import pprint
from pytorch_pretrained_bert.tokenization_gpt2 import GPT2Tokenizer
MEDIUM_MODEL = '../gpt2-pytorch-pretrained'
tokenizer = GPT2Tokenizer.from_pretrained(MEDIUM_MODEL)

sentence1 = 'However, it\'s too late'
sentence2 = 'However, it is too late'
ids_list1 = tokenizer.encode(sentence1)
ids_list2 = tokenizer.encode(sentence2)

sentence = [tokenizer.decoder[ids].replace('\u0120', '') for ids in ids_list2]
print(sentence)


# print(sentence.split())


from itertools import groupby
from string import punctuation
print(punctuation)


def sentence2gptlist(sentence):
    from itertools import groupby
    from string import punctuation

    sentence_list = [''.join(list(g)).strip() for k, g in groupby(sentence, key=lambda x: x.isalpha() or x=='\'')]
    print(sentence_list)
    new_list = []
    for ids, word in enumerate(sentence_list):
        if word not in punctuation and ids != 0:
            word = ' ' + word
        if word != '':
            new_list.append(word)

    return new_list


