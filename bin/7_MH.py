import torch
import random
import copy
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

gpus = [0, 1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
def compute_LL_onetoken(model, pre_tokens, next_token):
    pre_sentence = gpt2_tokenizer.decode(pre_tokens)
    logits = model.predict(pre_sentence)
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    return probabilities[next_token].tolist()
def sentence2gptlist(sentence):
    from itertools import groupby
    from string import punctuation

    sentence_list = [''.join(list(g)).strip() for k, g in groupby(sentence, key=lambda x: x.isalpha() or x == '\'')]
    new_list = []
    for ids, word in enumerate(sentence_list):
        if word not in punctuation and ids != 0:
            word = ' ' + word
        if word != '':
            new_list.append(word)
    print('new_list: ', new_list)
    return new_list
def compute_LL(model, sentence:str):
    # sentence_list = sentence2gptlist(sentence)
    # logLL = 0.0
    # # 从第二个词开始算句子的概率。
    # if len(sentence_list) != 0:
    #     pre_sentence = sentence_list[0]
    #     for word in sentence_list[1:]:
    #         prob = compute_LL_oneword(model, pre_sentence, word)
    #         logLL += math.log(prob)
    #         pre_sentence += word
    # assert pre_sentence == sentence
    tokens_list = gpt2_tokenizer.encode(sentence)
    logLL = 1.0
    # 从第二个词开始算句子的概率。
    if len(tokens_list) != 0:
        pre_token = tokens_list[0]
        for ids, token in enumerate(tokens_list):
            if ids > 0:
                prob = compute_LL_onetoken(model, tokens_list[:ids], token)
                logLL *= prob
    # assert tokens_list == sentence
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
            self.model = GPT2LMHeadModel.from_pretrained(MEDIUM_MODEL).cuda()
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

        inputs = torch.LongTensor([token_ids]).to(device)

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
# bert_path = '/root/xcn/bert_pytorch'
# bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
# bert_model = BertForMaskedLM.from_pretrained(bert_path)
# bert_token = bert_tokenizer.tokenize(sentence)



sentence = 'we types proposals'
keywords = gpt2_tokenizer.encode(sentence)
gpt2_token = [gpt2_tokenizer.decoder[ids].replace('\u0120', '')
              for ids in gpt2_tokenizer.encode(sentence)]



p_insert = 1/3
p_delete = 1/3
p_replace = 1/3


class PiOrigin:
    def LM_propility(self, x) -> float:
        return x
class PiKeyWords(PiOrigin):
    def __init__(self, keywords=[]):
        self.keywords = keywords
    def compute_pi(self, x:list=[]):
        for keyword in self.keywords:
            if keyword not in x:
                return 0
        return 1
class PiFactory:
    @staticmethod
    def typename(name):
        if name == 'keywords':
            return PiKeyWords()
        else:
            return None

pi = PiKeyWords(keywords)


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

def G_replace(x:str, pos, isInsert=False, isDelete=False, *ids):
    # ids的位置是根据tokenize后的位置计算的，而不是x的位置！
    # bert_tokens = bert_tokenizer.tokenize(x)
    gpt2_tokens = gpt2_tokenizer.encode(x)
    # print(gpt2_tokens)
    x_split = gpt2_tokenizer.decode(gpt2_tokens[:pos])
    # print(x_split)
    logits = model_345M.predict(x_split)
    candidate_logits, candidate_ids = logits.topk(10)
    candidate_ids = candidate_ids.tolist()
    if isDelete:
        assert(len(ids) == 1)
        candidate_ids = list(ids) + candidate_ids
    # print(candidate_ids)
    candidate_tokenslist = []
    if isInsert:
        gpt2_tokens.insert(pos, 0)
        for ids in candidate_ids:
            gpt2_tokens[pos] = ids
            candidate_tokenslist.append(copy.deepcopy(gpt2_tokens))
    else:
        for ids in candidate_ids:
            gpt2_tokens[pos] = ids
            candidate_tokenslist.append(copy.deepcopy(gpt2_tokens))
    # print(candidate_tokenslist)
    candidate_sentencelist = [gpt2_tokenizer.decode(x) for x in candidate_tokenslist]
    candidate_probilities = [compute_gptLL(gpt2_tokenizer.decode(x)) for x in candidate_tokenslist]
    candidate_probilities = [x/sum(candidate_probilities) for x in candidate_probilities]
    # print(candidate_sentencelist)
    # print(candidate_probilities)
    ret_index = random_index(candidate_probilities)

    return candidate_sentencelist[ret_index], candidate_probilities[ret_index], \
           candidate_probilities, candidate_tokenslist
def G_delete(x:str, pos):
    gpt2_tokens = gpt2_tokenizer.encode(x)
    deleted_ids = gpt2_tokens.pop(pos)
    sentence = gpt2_tokenizer.decode(gpt2_tokens)
    return sentence, deleted_ids

# print(G_delete(sentence, 3))
# G_replace(sentence, 3, True, True, 320)
def A_replace(x:str, idx) -> str:
    x_prop = G_replace(x, idx)[0]
    x_prop_tokens = gpt2_tokenizer.encode(x_prop)
    if pi.compute_pi(x_prop_tokens):
        return x_prop
    else:
        return x

def A_insert(x:str, idx) -> str:
    x_prop, g_prob, _, _ = G_replace(x, idx, isInsert=True)
    x_tokens = gpt2_tokenizer.encode(x)
    x_prop_tokens = gpt2_tokenizer.encode(x_prop)
    accept_prob = (p_delete/p_insert) * (compute_gptLL(x_prop)/compute_gptLL(x)) * (1/g_prob) \
                  * pi.compute_pi(x_prop_tokens) / pi.compute_pi(x_tokens)
    # print(accept_prob)

    randnum = random.random()
    if randnum < accept_prob:
        return x_prop
    else:
        return x

def A_delete(x:str, idx) -> str:
    x_prop, deleted_ids = G_delete(x, idx)
    x_tokens = gpt2_tokenizer.encode(x)
    x_prop_tokens = gpt2_tokenizer.encode(x_prop)
    accept_prob = (p_insert/p_delete) * (compute_gptLL(x_prop)/compute_gptLL(x)) * \
                  G_replace(x_prop, idx, True, True, deleted_ids)[2][0] * \
                  pi.compute_pi(x_prop_tokens) / pi.compute_pi(x_tokens)
    # print(accept_prob)
    randnum = random.random()
    if randnum < accept_prob:
        return x_prop
    else:
        return x

# ret = A_insert(sentence, 3)
# print(ret)


# 执行部分
proposal_prob = [1/3, 1/3, 1/3]

for _ in tqdm(range(100)):
    proposal = random_index(proposal_prob)
    gpt2_tokens = gpt2_tokenizer.encode(sentence)
    tokens_len = len(gpt2_tokens)
    position_prob = [1/tokens_len for _ in range(tokens_len)]
    position = random_index(position_prob)
    # 由于单向语言模型问题，Gibbs Sampling无法计算带suffix的LM概率，因此replace和insert操作都不能对第一个单词进行替换
    # 同时不能删除第一个节点，因为这样会输入空的tensor
    if proposal == 0 and position > 0:
        sentence = A_replace(sentence, position)
    elif proposal == 1:
        sentence = A_insert(sentence, position+1)
    elif proposal == 2 and position > 0:
        sentence = A_delete(sentence, position)
    print(sentence)


# 我觉得为了更好的效果需要重写candidate selector。