
import itertools

from pytorch_pretrained_bert.tokenization_gpt2 import GPT2Tokenizer
from pytorch_pretrained_bert.modeling_gpt2 import GPT2LMHeadModel
import torch
import tqdm
from typing import Dict, TypeVar, Generic, Optional
from collections import OrderedDict
import math

# from lm_explorer.lm.language_model import LanguageModel
# from lm_explorer.util.cache import LRUCache
# from lm_explorer.util.sampling import random_sample

K = TypeVar('K')
V = TypeVar('V')

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

class LanguageModel:
    def predict(self, previous: str, next: str) -> torch.Tensor:
        raise NotImplementedError

    def __getitem__(self, index: int) -> str:
        raise NotImplementedError


MEDIUM_MODEL = 'gpt2-pytorch-pretrained'

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

previous_str = 'However'
next_str = None
logits = model_345M.predict(previous_str, next_str)
probabilities = torch.nn.functional.softmax(logits)
best_logits, best_indices = logits.topk(10)
best_words = [model_345M[idx.item()] for idx in best_indices]
best_probabilities = probabilities[best_indices].tolist()

print('best_words: ', best_words)
print('best_probabilities: ', best_probabilities)
ids = model_345M.tokenizer.encode(' as')
print('best indices', best_indices)
print(probabilities[ids])

# print(probabilities)
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
    logLL = 0.0
    # 从第二个词开始算句子的概率。
    if len(sentence_list) != 0:
        pre_sentence = sentence_list[0]
        for word in sentence_list[1:]:
            prob = compute_LL_oneword(model, pre_sentence, word)
            logLL += math.log(prob)
            pre_sentence += word
    assert pre_sentence == sentence
    return logLL

print(compute_LL(model_345M, 'However, it may be too late'))
