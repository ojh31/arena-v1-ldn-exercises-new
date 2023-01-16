#%%
import functools
import json
import os
from typing import Any, List, Tuple, Union, Optional
import matplotlib.pyplot as plt
import torch
import torch as t
import torch.nn.functional as F
from fancy_einsum import einsum
from sklearn.linear_model import LinearRegression
from torch import nn
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from einops import rearrange, repeat
import pandas as pd
import numpy as np
import random

import w5d5_tests
from w5d5_transformer import ParenTransformer, SimpleTokenizer

MAIN = __name__ == "__main__"
DEVICE = t.device("cpu")
# %%
if MAIN:
    model = ParenTransformer(ntoken=5, nclasses=2, d_model=56, nhead=2, d_hid=56, nlayers=3).to(DEVICE)
    state_dict = t.load("w5d5_balanced_brackets_state_dict.pt")
    model.to(DEVICE)
    model.load_simple_transformer_state_dict(state_dict)
    model.eval()
    tokenizer = SimpleTokenizer("()")
    with open("w5d5_brackets_data.json") as f:
        data_tuples: List[Tuple[str, bool]] = json.load(f)
        print(f"loaded {len(data_tuples)} examples")
    assert isinstance(data_tuples, list)

class DataSet:
    '''A dataset containing sequences, is_balanced labels, and tokenized sequences'''

    def __init__(self, data_tuples: list):
        '''
        data_tuples is List[Tuple[str, bool]] signifying sequence and label
        '''
        self.strs = [x[0] for x in data_tuples]
        self.isbal = t.tensor([x[1] for x in data_tuples]).to(device=DEVICE, dtype=t.bool)
        self.toks = tokenizer.tokenize(self.strs).to(DEVICE)
        self.open_proportion = t.tensor([s.count("(") / len(s) for s in self.strs])
        self.starts_open = t.tensor([s[0] == "(" for s in self.strs]).bool()

    def __len__(self) -> int:
        return len(self.strs)

    def __getitem__(self, idx) -> Union["DataSet", tuple[str, t.Tensor, t.Tensor]]:
        if type(idx) == slice:
            return self.__class__(list(zip(self.strs[idx], self.isbal[idx])))
        return (self.strs[idx], self.isbal[idx], self.toks[idx])

    @property
    def seq_length(self) -> int:
        return self.toks.size(-1)

    @classmethod
    def with_length(cls, data_tuples: list[tuple[str, bool]], selected_len: int) -> "DataSet":
        return cls([(s, b) for (s, b) in data_tuples if len(s) == selected_len])

    @classmethod
    def with_start_char(cls, data_tuples: list[tuple[str, bool]], start_char: str) -> "DataSet":
        return cls([(s, b) for (s, b) in data_tuples if s[0] == start_char])

if MAIN:
    N_SAMPLES = 5000
    data_tuples = data_tuples[:N_SAMPLES]
    data = DataSet(data_tuples)
    print('examples: ', random.sample(data.strs, 5))
    print('isbal: ', pd.Series(data.isbal).value_counts())
    char_lens = [len(s) for s in data.strs]
    print('len: ', pd.Series(char_lens).value_counts().head())
    fig = px.histogram(x=char_lens)
    fig.update_layout(dict(
        xaxis_title='string length'
    ))
    fig.show()

# %%
def is_balanced_forloop(parens: str) -> bool:
    '''
    Return True if the parens are balanced.

    Parens is just the ( and ) characters, no begin or end tokens.
    '''
    level = 0
    for c in parens:
        if c == '(':
            level += 1
        elif c == ')':
            level -= 1
        else:
            raise ValueError(f'Bad character in parens str: {c}')
        if level < 0:
            return False
    return level == 0

if MAIN:
    examples = [
        "()", 
        "))()()()()())()(())(()))(()(()(()(", 
        "((()()()()))", 
        "(()()()(()(())()", 
        "()(()(((())())()))"
    ]
    labels = [True, False, True, False, True]
    for (parens, expected) in zip(examples, labels):
        actual = is_balanced_forloop(parens)
        assert expected == actual, f"{parens}: expected {expected} got {actual}"
    print("is_balanced_forloop ok!")
# %%
def is_balanced_vectorized(tokens: t.Tensor) -> bool:
    '''
    tokens: 
        sequence of tokens including begin, end and pad tokens 
        - recall that 3 is '(' and 4 is ')'
    '''
    level_diffs = t.where(
        tokens == 3,
        1,
        t.where(tokens == 4, -1, 0)
    )
    levels = level_diffs.cumsum(dim=-1)
    return (levels >= 0).all() and levels[-1] == 0

if MAIN:
    for (tokens, expected) in zip(tokenizer.tokenize(examples), labels):
        actual = is_balanced_vectorized(tokens)
        assert expected == actual, f"{tokens}: expected {expected} got {actual}"
    print("is_balanced_vectorized ok!")
# %%
