import torch
import random
from string import punctuation
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

import itertools

from pytorch_pretrained_bert.tokenization_gpt2 import GPT2Tokenizer
from pytorch_pretrained_bert.modeling_gpt2 import GPT2LMHeadModel
import torch
from tqdm import tqdm
from typing import Dict, TypeVar, Generic, Optional
from collections import OrderedDict
import math

K = TypeVar('K')
V = TypeVar('V')
MEDIUM_MODEL = '../gpt2-pytorch-pretrained'


class LRUCache(Generic[K, V]):
    def __init__(self, capacity: int, default_value = None):
        self._capacity = capacity
        self._cache: Dict[K, V] = OrderedDict()
        self._default_value = default_value

    def __getitem__(self, key: K) -> Optional[V]:
        if self._capacity == 0:
            return self._default_value
        try:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        except KeyError:
            return self._default_value

    def __setitem__(self, key: K, value: V) -> None:
        if self._capacity == 0:
            return

        try:
            self._cache.pop(key)
        except KeyError:
            if len(self._cache) >= self._capacity:
                self._cache.popitem(last=False)
        self._cache[key] = value

def random_sample(logits: torch.Tensor, temperature: float = 1.0) -> int:
    d = torch.distributions.Categorical(logits=logits / temperature)
    return d.sample().item()
def compute_LL_oneword(model, pre_sentence, next_word):
    logits = model.predict(pre_sentence)
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    ids = model.tokenizer.encode(next_word)
    return probabilities[ids].tolist()[0]
def sentence2gptlist(sentence):
    from itertools import groupby
    from string import punctuation

    sentence_list = [''.join(list(g)).strip() for k, g in groupby(sentence, key=lambda x: x.isalpha() or x == '\'')]
    print(sentence_list)
    new_list = []
    for ids, word in enumerate(sentence_list):
        if word not in punctuation and ids != 0:
            word = ' ' + word
        if word != '':
            new_list.append(word)

    return new_list
def compute_LL(model, sentence:str):
    sentence_list = sentence2gptlist(sentence)
    # logLL = 0.0
    # # 从第二个词开始算句子的概率。
    # if len(sentence_list) != 0:
    #     pre_sentence = sentence_list[0]
    #     for word in sentence_list[1:]:
    #         prob = compute_LL_oneword(model, pre_sentence, word)
    #         logLL += math.log(prob)
    #         pre_sentence += word
    # assert pre_sentence == sentence
    logLL = 1.0
    # 从第二个词开始算句子的概率。
    if len(sentence_list) != 0:
        pre_sentence = sentence_list[0]
        for word in sentence_list[1:]:
            prob = compute_LL_oneword(model, pre_sentence, word)
            logLL *= prob
            pre_sentence += word
    assert pre_sentence == sentence
    return logLL
def compute_gptLL(sentence:str):
    return compute_LL(model_345M, sentence)

class LanguageModel:
    def predict(self, previous: str, next: str) -> torch.Tensor:
        raise NotImplementedError

    def __getitem__(self, index: int) -> str:
        raise NotImplementedError
class GPT2LanguageModel(LanguageModel):
    def __init__(self, cache_size: int = 0, model_name: str = '345M') -> None:
        """
        Each cache element is about 8MB, so size accordingly.
        """
        # Cache stores tuples, so default value is a tuple
        self._cache = LRUCache(cache_size, default_value=(None, None))
        self.tokenizer = GPT2Tokenizer.from_pretrained(MEDIUM_MODEL)
        if model_name == '117M':
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        elif model_name == '345M':
            self.model = GPT2LMHeadModel.from_pretrained(MEDIUM_MODEL)
        else:
            exit("model name not found")

        # The end of text marker.
        self.END_OF_TEXT = self.tokenizer.encoder["<|endoftext|>"]

    def predict(self, previous: str, next: str = None) -> torch.Tensor:

        past_logits, past = self._cache[previous]

        # CASE 1: Previously seen input, no next
        if next is None and past is not None:
            return past_logits

        # CASE 2: Previously seen input, yes next
        elif past is not None:
            token_ids = self.tokenizer.encode(next)
        # CASE 3: Brand new input, no next
        elif next is None:
            token_ids = self.tokenizer.encode(previous)
        # CASE 4: Brand new input, yes next
        else:
            token_ids = self.tokenizer.encode(previous) + self.tokenizer.encode(next)

        inputs = torch.LongTensor([token_ids])

        logits, present = self.model(inputs, past=past)
        logits = logits[0, -1]

        key = previous if next is None else previous + next
        self._cache[key] = logits, present

        return logits

    def __getitem__(self, index: int) -> str:
        return self.tokenizer.decode([index])

    def generate(self, seed: str = "", max_len: int = None) -> str:

        output = seed
        logits = self.predict(seed)

        if max_len is None:
            it = tqdm.tqdm(itertools.count())
        else:
            it = tqdm.trange(max_len)

        for _ in it:
            next_id = random_sample(logits)
            next_word = self[next_id]

            print(next_word)

            if next_word == "<|endoftext|>":
                break

            logits = self.predict(output, next_word)
            output += next_word

        return output
model_345M = GPT2LanguageModel(model_name='345M')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(MEDIUM_MODEL)

# bert拜拜 你真不好用
bert_path = '/root/xcn/bert_pytorch'
bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
bert_model = BertForMaskedLM.from_pretrained(bert_path)



sentence = 'However, it is too late'
bert_token = bert_tokenizer.tokenize(sentence)
gpt2_token = [gpt2_tokenizer.decoder[ids].replace('\u0120', '')
              for ids in gpt2_tokenizer.encode(sentence)]
print(bert_token)
print(gpt2_token)


p_insert = 1/3
p_delete = 1/3
p_replace = 1/3


class PiOrigin:
    def LM_propility(self, x) -> float:
        return x
class PiKeyWords(PiOrigin):
    def compute_pi(self, x, keywords:list=[]):
        for keyword in keywords:
            if keyword not in self.x:
                return 0.0
            else:
                return self.LM_propility(self.x)
class PiFactory:
    @staticmethod
    def typename(name):
        if name == 'keywords':
            return PiKeyWords()
        else:
            return None


pi_keywords = PiFactory.typename('keywords')


def random_index(prob_list:list) -> int:
    # 按概率返回一个随机的index
    start = 0
    index = 0
    randnum = random.random()

    for index, scope in enumerate(prob_list):
        start += scope
        if randnum <= start:
            break
    return index

def Candidate_Generator():
    pass

def G_replace(x:str, ids, isInsert=False):
    # ids的位置是根据tokenize后的位置计算的，而不是x的位置！
    bert_tokens = bert_tokenizer.tokenize(x)
    if isInsert:
        bert_tokens.insert(ids, '[MASK]')
    else:
        bert_tokens[ids] = '[MASK]'

    tokens_ids = bert_tokenizer.convert_tokens_to_ids(bert_tokens)
    segments_ids = [0] * len(tokens_ids)

    tokens_tensor = torch.tensor([tokens_ids])
    segments_tensor = torch.tensor([segments_ids])

    bert_predictions = bert_model(tokens_tensor, segments_tensor)
    bert_probilities = torch.nn.functional.softmax(bert_predictions[0, ids], dim=-1)
    ret_index = random_index(bert_probilities.tolist())
    tokens_ids[ids] = ret_index

    sentence_list = bert_tokenizer.convert_ids_to_tokens(tokens_ids)
    new_list = []
    for ids, word in enumerate(sentence_list):
        if word not in punctuation and ids != 0:
            word = ' ' + word
        if word != '':
            new_list.append(word)
    sentence = ''.join(new_list)
    return sentence, bert_probilities[ret_index], bert_probilities
def G_delete(x:str, idx):
    bert_tokens = bert_tokenizer.tokenize(x)
    deleted_word = bert_tokens.pop(idx)
    sentence_list = bert_tokens
    new_list = []
    for ids, word in enumerate(sentence_list):
        if word not in punctuation and ids != 0:
            word = ' ' + word
        if word != '':
            new_list.append(word)
    sentence = ''.join(new_list)
    deleted_ids = bert_tokenizer.convert_tokens_to_ids([deleted_word])
    return sentence, deleted_ids

# print(G_delete(sentence, 3))
# G_replace(sentence, 3)
def A_replace(x:str, idx) -> str:
    return G_replace(x, idx)[0]

def A_insert(x:str, idx) -> str:
    x_prop, g_prob, _ = G_replace(x, idx, isInsert=True)
    accept_prob = (p_delete/p_insert) * (compute_gptLL(x_prop)/compute_gptLL(x)) * (1/g_prob)
    # print(accept_prob)

    randnum = random.random()
    if randnum < accept_prob:
        return x_prop
    else:
        return x

def A_delete(x:str, idx) -> str:
    x_prop, deleted_ids = G_delete(x, idx)
    #  这个LM模型取了对数，绝对值越小概率越高，所以第二项取个倒
    accept_prob = (p_insert/p_delete) * (compute_gptLL(x_prop)/compute_gptLL(x)) * \
                  G_replace(x_prop, idx, isInsert=True)[2][deleted_ids]
    # print(accept_prob)
    randnum = random.random()
    if randnum < accept_prob:
        return x_prop
    else:
        return x

# ret = A_insert(sentence, 3)
# print(ret)

proposal_prob = [1/3, 1/3, 1/3]

for _ in tqdm(range(200)):
    proposal = random_index(proposal_prob)
    bert_tokens = bert_tokenizer.tokenize(sentence)
    tokens_len = len(bert_tokens)
    position_prob = [1/tokens_len for _ in range(tokens_len)]
    position = random_index(position_prob)
    # 由于单向语言模型问题，Gibbs Sampling无法计算带suffix的LM概率，因此replace和insert操作都不能对第一个单词进行替换
    # delete这种憨憨操作就不存在这个问题
    if proposal == 0 and position > 0:
        sentence = A_replace(sentence, position)
    elif proposal == 1:
        sentence = A_insert(sentence, position+1)
    elif proposal == 2:
        sentence = A_delete(sentence, position)

print(sentence)