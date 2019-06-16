
import itertools

from pytorch_pretrained_bert.tokenization_gpt2 import GPT2Tokenizer
from pytorch_pretrained_bert.modeling_gpt2 import GPT2LMHeadModel
import torch
import tqdm
from typing import Dict, TypeVar, Generic, Optional
from collections import OrderedDict

MEDIUM_MODEL = 'https://storage.googleapis.com/allennlp/models/gpt2-345M-dump'


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
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir='/Users/xcn/Desktop/gpt-2/')
        if model_name == '117M':
            self.model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir='/Users/xcn/Desktop/gpt-2/')
        elif model_name == '345M':
            self.model = GPT2LMHeadModel.from_pretrained(MEDIUM_MODEL, cache_dir='/Users/xcn/Desktop/gpt-2/')
        else:
            exit("model name not found")

        # The end of text marker.
        self.END_OF_TEXT = self.tokenizer.encoder["<|endoftext|>"]

model = GPT2LanguageModel()