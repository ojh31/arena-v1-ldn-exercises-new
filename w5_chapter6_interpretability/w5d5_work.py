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
    model = ParenTransformer(
        ntoken=5, nclasses=2, d_model=56, nhead=2, d_hid=56, nlayers=3
    ).to(DEVICE)
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
if MAIN:
    toks = tokenizer.tokenize(examples).to(DEVICE)
    out = model(toks)
    prob_balanced = out.exp()[:, 1]
    print("Model confidence:\n" + "\n".join([f"{ex:34} : {prob:.4%}" for ex, prob in zip(examples, prob_balanced)]))

def run_model_on_data(model: ParenTransformer, data: DataSet, batch_size: int = 200) -> t.Tensor:
    '''Return probability that each example is balanced'''
    ln_probs = []
    for i in range(0, len(data.strs), batch_size):
        toks = data.toks[i : i + batch_size]
        with t.no_grad():
            out = model(toks)
        ln_probs.append(out)
    out = t.cat(ln_probs).exp()
    assert out.shape == (len(data), 2)
    return out

if MAIN:
    test_set = data
    n_correct = t.sum((run_model_on_data(model, test_set).argmax(-1) == test_set.isbal).int())
    print(f"\nModel got {n_correct} out of {len(data)} training examples correct!")
#%% [markdown]
#### Part 2: Moving backwards
# %%
def get_post_final_ln_dir(model: ParenTransformer) -> t.Tensor:
    '''
    Use the weights of the final linear layer (model.decoder) to 
    identify the direction in the space that goes into the linear layer (and out of the LN) 
    corresponding to an 'unbalanced' classification. 
    Hint: this is a one line function.

    out: shape (e=56, )
    '''
    return (model.decoder.weight[0, :] - model.decoder.weight[1, :])
# %%
def get_inputs(model: ParenTransformer, data: DataSet, module: nn.Module) -> t.Tensor:
    '''
    Get the inputs to a particular submodule of the model when run on the data.
    Returns a tensor of size (data_pts, seq_pos, emb_size).
    '''
    inputs = []
    def hook(hook_module: nn.Module, input: Tuple, output: t.Tensor) -> None:
        assert hook_module == module
        for inp in input:
            inputs.append(inp)

    handle = module.register_forward_hook(hook)
    run_model_on_data(model, data)
    handle.remove()
    return t.cat(inputs)

def get_outputs(model: ParenTransformer, data: DataSet, module: nn.Module) -> t.Tensor:
    '''
    Get the outputs from a particular submodule of the model when run on the data.
    Returns a tensor of size (data_pts, seq_pos, emb_size).
    '''
    outputs = []
    def hook(hook_module: nn.Module, input: Tuple, output: t.Tensor) -> None:
        assert hook_module == module
        outputs.append(output)

    handle = module.register_forward_hook(hook)
    run_model_on_data(model, data)
    handle.remove()
    return t.cat(outputs)

if MAIN:
    w5d5_tests.test_get_inputs(get_inputs, model, data)
    w5d5_tests.test_get_outputs(get_outputs, model, data)
# %%
def get_ln_fit(
    model: ParenTransformer, data: DataSet, ln_module: nn.LayerNorm, seq_pos: Union[None, int]
) -> Tuple[LinearRegression, t.Tensor]:
    '''
    if seq_pos is None, find best fit for all sequence positions. 
    Otherwise, fit only for given seq_pos.

    Returns: A tuple of 
        - a (fitted) sklearn LinearRegression object
        - a dimensionless tensor containing the r^2 of the fit (hint: 
            wrap a value in torch.tensor() to make a dimensionless tensor)
    '''
    inputs = get_inputs(model, data, ln_module)
    outputs = get_outputs(model, data, ln_module)
    if seq_pos is not None:
        inputs = inputs[:, seq_pos, :]
        outputs = outputs[:, seq_pos, :]
    lr = LinearRegression()
    lr.fit(inputs, outputs)
    r2 = t.tensor(lr.score(inputs, outputs))
    return (lr, r2)


if MAIN:
    (final_ln_fit, r2) = get_ln_fit(model, data, model.norm, seq_pos=0)
    print("r^2: ", r2)
    w5d5_tests.test_final_ln_fit(model, data, get_ln_fit)
# %%
def get_pre_final_ln_dir(model: ParenTransformer, data: DataSet) -> t.Tensor:
    '''
    x_norm -> (ln) x_linear -> (linear and softmax) logit_diff
    logit_diff is maximised by post_final_ln_dir: shape (e=56, )
    post_final_ln_dir is maximised by ... in seq_pos=0

    pre_final_ln_dir: shape (e, ) = (56, )
    L: shape (e, e) = (56, 56)
    post_final_ln_dir: shape (e, ) = (56, )
    '''
    post_final_ln_dir = get_post_final_ln_dir(model)
    ln_fit, _ = get_ln_fit(model, data, model.norm, seq_pos=0)
    L = t.tensor(ln_fit.coef_)
    pre_final_ln_dir = post_final_ln_dir @ L
    return pre_final_ln_dir


if MAIN:
    w5d5_tests.test_pre_final_ln_dir(model, data, get_pre_final_ln_dir)
# %%
def get_out_by_head(
    model: ParenTransformer, data: DataSet, layer: int
) -> t.Tensor:
    '''
    Get the output of the heads in a particular layer when the model is run on the data.
    Returns a tensor of shape (batch, num_heads, seq, embed_width)
    '''
    W_O = model.layers[layer].self_attn.W_O
    r = get_inputs(model, data, W_O)
    r_block = rearrange(
        r, 
        'b s (n h)-> b n s h', 
        n=model.nhead,
    ) # split up the rows of r
    W_block = rearrange(
        W_O.weight, 
        'e (n h) -> n e h', 
        n=model.nhead,
    ) # split up the columns of W_O
    out = einsum(
        'n e h, b n s h-> b n s e', 
        W_block, 
        r_block,
    ) # The common dim in the matmuls here is h
    return out

if MAIN:
    w5d5_tests.test_get_out_by_head(get_out_by_head, model, data)
# %%
def get_out_by_components(model: ParenTransformer, data: DataSet) -> t.Tensor:
    '''
    Computes a tensor of shape [10, dataset_size, seq_pos, emb] representing 
    the output of the model's components when run on the data.

    The first dimension is [
        embeddings, head 0.0, head 0.1, mlp 0, head 1.0, head 1.1, mlp 1, head 2.0, head 2.1, mlp 2
    ]
    '''
    embeddings = get_outputs(model, data, model.pos_encoder).unsqueeze(dim=1)
    output = [embeddings]
    for i, layer in enumerate(model.layers):
        output.append(get_out_by_head(model, data, i))
        output.append(get_outputs(model, data, layer.linear2).unsqueeze(dim=1))
    o_tensor = t.cat(output, dim=1)
    o_tensor = rearrange(o_tensor, 'b c s e -> c b s e')
    return o_tensor

if MAIN:
    w5d5_tests.test_get_out_by_component(get_out_by_components, model, data)
# %%
def hists_per_comp(magnitudes, data, n_layers=3, xaxis_range=(-1, 1)):
    num_comps = magnitudes.shape[0]
    titles = {
        (1, 1): "embeddings",
        (2, 1): "head 0.0",
        (2, 2): "head 0.1",
        (2, 3): "mlp 0",
        (3, 1): "head 1.0",
        (3, 2): "head 1.1",
        (3, 3): "mlp 1",
        (4, 1): "head 2.0",
        (4, 2): "head 2.1",
        (4, 3): "mlp 2"
    }
    assert num_comps == len(titles)

    fig = make_subplots(rows=n_layers+1, cols=3)
    for ((row, col), title), mag in zip(titles.items(), magnitudes):
        if row == n_layers+2: break
        fig.add_trace(
            go.Histogram(
                x=mag[data.isbal].numpy(), name="Balanced", marker_color="blue", opacity=0.5, 
                legendgroup = '1', showlegend=title=="embeddings"
            ), 
            row=row, 
            col=col)
        fig.add_trace(
            go.Histogram(
                x=mag[~data.isbal].numpy(), name="Unbalanced", marker_color="red", opacity=0.5, 
                legendgroup = '2', showlegend=title=="embeddings"
            ), 
            row=row, 
            col=col
        )
        fig.update_xaxes(title_text=title, row=row, col=col, range=xaxis_range)
    fig.update_layout(
        width=1200, height=250*(n_layers+1), barmode="overlay", 
        legend=dict(yanchor="top", y=0.92, xanchor="left", x=0.4), 
        title="Histograms of component significance"
    )
    fig.show()

if MAIN:
    '''
    magnitudes: shape [10, dataset_size]
    dot product of the component's output with the unbalanced direction on this sample
    normalize it by subtracting the mean of the dot product of this component's output with 
    the unbalanced direction on balanced samples
    '''
    out_by_comp = get_out_by_components(model, data) # shape [10, dataset_size, seq_pos, emb]
    zero_by_comp = out_by_comp[:, :, 0, :].squeeze() # shape [10, dataset_size, emb]
    pre_final_ln_dir = get_pre_final_ln_dir(model, data) # shape [emb]
    pre_final_ln_mags = einsum(
        'c d e, e -> c d', zero_by_comp, pre_final_ln_dir
    ) # shape [10, dataset_size]
    pre_final_ln_means = pre_final_ln_mags[:, data.isbal].mean(axis=1)
    pre_final_ln_mags = (pre_final_ln_mags.T - pre_final_ln_means.T).T.detach()
    hists_per_comp(pre_final_ln_mags, data, xaxis_range=[-10, 20])
# %%
if MAIN:
    # We read right to left and want 4s to have a matching 3
    token_arr = t.flip(data.toks, (1, )) # flip seq dim
    elevation_diffs = t.where(
        token_arr == 4,
        1,
        t.where(token_arr == 3, -1, 0)
    )
    elevation = elevation_diffs.cumsum(dim=-1)
    negative_failure = (elevation < 0).any(dim=1) # shape [dataset,]
    total_elevation_failure = elevation[:, -1] != 0 # shape [dataset,]
    h20_in_d = pre_final_ln_mags[-3, :]
    h21_in_d = pre_final_ln_mags[-2, :]

    failure_types = np.full(len(h20_in_d), "", dtype=np.dtype("U32"))
    failure_types_dict = {
        "both failures": negative_failure & total_elevation_failure,
        "just neg failure": negative_failure & ~total_elevation_failure,
        "just total elevation failure": ~negative_failure & total_elevation_failure,
        "balanced": ~negative_failure & ~total_elevation_failure
    }
    for name, mask in failure_types_dict.items():
        failure_types = np.where(mask, name, failure_types)
    failures_df = pd.DataFrame({
        "Head 2.0 contribution": h20_in_d,
        "Head 2.1 contribution": h21_in_d,
        "Failure type": failure_types
    })[data.starts_open.tolist()]
    fig = px.scatter(
        failures_df, 
        x="Head 2.0 contribution", y="Head 2.1 contribution", color="Failure type", 
        title="h20 vs h21 for different failure types", template="simple_white", height=600, width=800,
        category_orders={"color": failure_types_dict.keys()}
    ).update_traces(marker_size=4)
    fig.show()
# %%
if MAIN:
    fig = px.scatter(
        x=data.open_proportion, y=h20_in_d, color=failure_types, 
        title="Head 2.0 contribution vs proportion of open brackets '('", template="simple_white", height=500, width=800,
        labels={"x": "Open-proportion", "y": "Head 2.0 contribution"}, category_orders={"color": failure_types_dict.keys()}
    ).update_traces(marker_size=4, opacity=0.5).update_layout(legend_title_text='Failure type')
    fig.show()
# %% [markdown]
#### Part 3: Total Elevation Circuit
#%%
def get_attn_probs(
    model: ParenTransformer, tokenizer: SimpleTokenizer, data: DataSet, 
    layer: int, head: int,
) -> t.Tensor:
    '''
    Returns: (N_SAMPLES, max_seq_len, max_seq_len) tensor that 
    sums to 1 over the last dimension.
    '''
    attn_layer = model.layers[layer].self_attn
    attention_inputs = get_inputs(
        model, data, attn_layer
    ) # shape (batch, seq, hidden_size)
    attn_scores = attn_layer.attention_pattern_pre_softmax(
        attention_inputs
    ) # shape [b, n, s, s]
    padding_mask = data.toks == tokenizer.PAD_TOKEN
    additive_mask = t.where(padding_mask, -10000, 0)[:, None, None, :] 
    attn_scores += additive_mask
    attention_probabilities = attn_scores.softmax(
        dim=-1
    )[:, head, :, :].squeeze() # shape [b, s, s]
    return attention_probabilities.detach()


if MAIN:
    attn_probs = get_attn_probs(model, tokenizer, data, 2, 0)
    attn_probs_open = attn_probs[data.starts_open].mean(0)[[0]]
    px.bar(
        y=attn_probs_open.squeeze().numpy(), 
        labels={"y": "Probability", "x": "Key Position"},
        template="simple_white", 
        height=500, 
        width=600, 
        title="Avg Attention Probabilities for '(' query from query 0",
    ).update_layout(showlegend=False, hovermode='x unified').show()
# %%
def get_WV(model: ParenTransformer, layer: int, head: int) -> t.Tensor:
    '''
    Returns the W_V matrix of a head. 
    Should be a CPU tensor of size (d_model / num_heads, d_model)
    '''
    w_v = model.layers[layer].self_attn.W_V.weight # shape [nh, nh]
    head_size = model.d_model // model.nhead
    head_indices = np.arange(head * head_size, (head + 1) * head_size)
    return w_v[head_indices, :].detach().to(device='cpu')

def get_WO(model: ParenTransformer, layer: int, head: int) -> t.Tensor:
    '''
    Returns the W_O matrix of a head. 
    Should be a CPU tensor of size (d_model, d_model / num_heads)
    '''
    w_o = model.layers[layer].self_attn.W_O.weight
    head_size = model.d_model // model.nhead
    head_indices = np.arange(head * head_size, (head + 1) * head_size)
    return w_o[:, head_indices].detach().to(device='cpu')

def get_WOV(model: ParenTransformer, layer: int, head: int) -> t.Tensor:
    '''
    out: shape [d_model, d_model]
    '''
    return get_WO(model, layer, head) @ get_WV(model, layer, head)

def get_pre_20_dir(model, data):
    '''
    Returns the direction propagated back through the OV matrix of 2.0 and 
    then through the layernorm before the layer 2 attention heads.
    out: numpy array shape [nh]
    '''
    pre_final_ln_dir = get_pre_final_ln_dir(model, data).detach() # shape [nh]
    ln_fit, _ = get_ln_fit(model, data, model.layers[2].norm1, seq_pos=1)
    L = ln_fit.coef_ # shape [nh, nh]
    w_ov = get_WOV(model, layer=2, head=0).detach() # shape [nh, nh]
    return (pre_final_ln_dir @ w_ov @ L) # shape [nh]

if MAIN:
    w5d5_tests.test_get_WV(model, get_WV)
    w5d5_tests.test_get_WO(model, get_WO)
    w5d5_tests.test_get_pre_20_dir(model, data, get_pre_20_dir)

if MAIN:
    out_by_comp = get_out_by_components(model, data) # shape [10, dataset_size, seq_pos, emb]
    zero_by_comp = out_by_comp[:, :, 0, :].squeeze() # shape [10, dataset_size, emb]
    pre_20_dir = get_pre_20_dir(model, data) # shape [emb]
    pre_20_mags = einsum(
        'c d e, e -> c d', zero_by_comp, pre_20_dir
    ) # shape [10, dataset_size]
    pre_20_means = pre_20_mags[:, data.isbal].mean(axis=1)
    pre_20_mags = (pre_20_mags.T - pre_20_means.T).T.detach()
    hists_per_comp(pre_20_mags, data, n_layers=2, xaxis_range=(-7, 7))
# %%
def mlp_attribution_scatter(magnitudes, data, failure_types):
    for layer in range(2):
        fig = px.scatter(
            x=data.open_proportion[data.starts_open], y=magnitudes[3+layer*3, data.starts_open], 
            color=failure_types[data.starts_open], category_orders={"color": failure_types_dict.keys()},
            title=f"Amount MLP {layer} writes in unbalanced direction for Head 2.0", 
            template="simple_white", height=500, width=800,
            labels={"x": "Open-proportion", "y": "Head 2.0 contribution"}
        ).update_traces(marker_size=4, opacity=0.5).update_layout(legend_title_text='Failure type')
        fig.show()

if MAIN:
    mlp_attribution_scatter(pre_20_mags, data, failure_types)
# %%
def out_by_neuron(model: ParenTransformer, data: DataSet, layer: int):
    '''
    Return shape: [len(data), seq_len, neurons, out]
    '''
    x = get_inputs(
        model, data, model.layers[layer].linear1
    ) # shape [b, s, d_model]
    B = model.layers[layer].linear1.weight # shape [d_hid, d_mod]
    c = model.layers[layer].linear1.bias # shape [d_hid]
    A = model.layers[layer].linear2.weight # shape [d_mod, d_hid]
    d = model.layers[layer].linear2.bias # shape [d_mod]
    f = model.layers[layer].activation
    Bx = einsum('h m, b s m -> b s h', B, x)
    f_Bx_c = f(Bx + c) # shape [b, s, d_hid]
    f_Bx_c_A = einsum('b s h, m h -> b s h m', f_Bx_c, A)
    return f_Bx_c_A

# @functools.cache
def out_by_neuron_in_20_dir(
    model: ParenTransformer, data: DataSet, layer: int,
):
    neuron_outs = out_by_neuron(model, data, layer)
    pre_20_dir = get_pre_20_dir(model, data)
    dot = einsum('b s h m, m -> b s h', neuron_outs, pre_20_dir)
    return dot
#%%
def plot_neurons(model, data, failure_types, layer):
    # Get neuron significances for head 2.0, sequence position #1 output
    neurons_in_d = out_by_neuron_in_20_dir(model, data, layer)[data.starts_open, 1, :].detach()

    # Get data that can be turned into a dataframe (plotly express is sometimes easier to use with a dataframe)
    # Plot a scatter plot of all the neuron contributions, color-coded according to failure type, with slider to view neurons
    neuron_numbers = repeat(t.arange(model.d_model), "n -> (s n)", s=data.starts_open.sum())
    failure_types = repeat(failure_types[data.starts_open], "s -> (s n)", n=model.d_model)
    data_open_proportion = repeat(data.open_proportion[data.starts_open], "s -> (s n)", n=model.d_model)
    df = pd.DataFrame({
        "Output in 2.0 direction": neurons_in_d.flatten(),
        "Neuron number": neuron_numbers,
        "Open-proportion": data_open_proportion,
        "Failure type": failure_types
    })
    px.scatter(
        df, 
        x="Open-proportion", y="Output in 2.0 direction", 
        color="Failure type", 
        animation_frame="Neuron number",
        title=f"Neuron contributions from layer {layer}", 
        template="simple_white", 
        height=500, 
        width=800
    ).update_traces(marker_size=3).update_layout(
        xaxis_range=[0, 1], yaxis_range=[-5, 5]
    ).show()

    # Work out the importance (average difference in unbalanced contribution between 
    # balanced and inbalanced dirs) for each neuron
    # Plot a bar chart of the per-neuron importances
    neuron_importances = neurons_in_d[~data.isbal[data.starts_open]].mean(0) - neurons_in_d[data.isbal[data.starts_open]].mean(0)
    px.bar(
        x=t.arange(model.d_model), 
        y=neuron_importances, 
        title=f"Importance of neurons in layer {layer}", 
        labels={"x": "Neuron number", "y": "Mean contribution in unbalanced dir"},
        template="simple_white", 
        height=400, 
        width=600, 
        hover_name=t.arange(model.d_model), 
        # hovermode="x unified"
    ).show()

if MAIN:
    for layer in range(2):
        plot_neurons(model, data, failure_types, layer)
# %%
def get_Q_and_K(
    model: ParenTransformer, layer: int, head: int
) -> Tuple[t.Tensor, t.Tensor]:
    '''
    Get the Q and K weight matrices for the attention head at the given indices.

    Return: Tuple of two tensors, both with shape (embedding_size, head_size)
    '''
    head_size = model.d_model // model.nhead
    head_indices = np.arange(head * head_size, (head + 1) * head_size)
    attn = model.layers[layer].self_attn
    q = attn.W_Q.weight[head_indices, :].T
    k = attn.W_K.weight[head_indices, :].T
    return (q, k)


def qk_calc_termwise(
    model: ParenTransformer, layer: int, head: int, 
    q_embedding: t.Tensor, k_embedding: t.Tensor
) -> t.Tensor:
    '''
    Get the pre-softmax attention scores that would be calculated by 
    the given attention head from the given embeddings.

    q_embedding: tensor of shape (seq_len, embedding_size)
    k_embedding: tensor of shape (seq_len, embedding_size)

    Returns: tensor of shape (seq_len, seq_len)
    '''
    q, k = get_Q_and_K(model, layer, head)
    qx = einsum('e h, s e -> s h', q, q_embedding)
    kx = einsum('e h, s e -> s h', k, k_embedding)

    a = einsum(
        "seqQ h, seqK h -> seqQ seqK", 
        qx, 
        kx,
    ) / (q.shape[-1] ** 0.5)
    return a


if MAIN:
    w5d5_tests.qk_test(model, get_Q_and_K)
    w5d5_tests.test_qk_calc_termwise(model, tokenizer, qk_calc_termwise)
# %%
def embedding(model: ParenTransformer, tokenizer: SimpleTokenizer, char: str) -> torch.Tensor:
    '''
    out: shape [1, emb_dim]
    '''
    assert char in ("(", ")")
    token = tokenizer.t_to_i[char]
    tokens = t.tensor([token])
    emb = model.encoder(tokens)
    return emb

if MAIN:
    w5d5_tests.embedding_test(model, tokenizer, embedding)
# %%
if MAIN:
    open_emb = embedding(model, tokenizer, "(")
    closed_emb = embedding(model, tokenizer, ")")
    pos_embeds = model.pos_encoder.pe
    open_emb_ln_per_seqpos = model.layers[0].norm1(open_emb.to(DEVICE) + pos_embeds[1:41])
    close_emb_ln_per_seqpos = model.layers[0].norm1(closed_emb.to(DEVICE) + pos_embeds[1:41])
    attn_score_open_avg = qk_calc_termwise(
        model, 
        layer=0, 
        head=0, 
        q_embedding=open_emb_ln_per_seqpos, 
        k_embedding=0.5 * (open_emb_ln_per_seqpos + close_emb_ln_per_seqpos),
    )
    attn_prob_open = attn_score_open_avg.softmax(-1).detach().clone().numpy()
    px.imshow(
        attn_prob_open, 
        color_continuous_scale="RdBu_r", height=500, width=550,
        labels={"x": "Key Position", "y": "Query Position", "color": "Attn prob"},
        title="Predicted Attention Probabilities for '(' query", origin="lower"
    ).update_layout(margin=dict(l=60, r=60, t=80, b=40)).show()
# %%
def avg_attn_probs_0_0(
    model: ParenTransformer, data: DataSet, tokenizer: SimpleTokenizer, query_token: int
) -> t.Tensor:
    '''
    Calculate the average attention probs for the 0.0 attention head for 
    the provided data when the query is the given query token.
    That is, q_embedding is given and k_embedding is empirical
    We want to take the (b, s) pairs for which the query token is query_token.
    For this, we need the (b, s, n, h) query embedding
    Returns a tensor of shape (seq, seq)
    '''
    q_tensor = t.tensor(query_token)
    attn_layer = model.layers[layer].self_attn
    x = get_inputs(
        model, data, attn_layer
    ) # shape (batch, seq, hidden_size)
    all_attn_probs = get_attn_probs(
        model, tokenizer, data, layer=0, head=0
    ) # shape [b, sQ, sK], need to subset sQ and then merge into batch dim
    _, _, sK = all_attn_probs.shape
    seq_q_mask = data.toks == query_token # shape [b, sQ]
    seq_q_mask = repeat(seq_q_mask, 'b sQ -> b sQ sK', sK=sK)
    masked_attn_probs = t.where(seq_q_mask, all_attn_probs, 0) # shape [b, s, s]
    avg_probs = masked_attn_probs.mean(dim=0) # shape [s, s]
    return avg_probs.detach()



if MAIN:
    data_len_40 = DataSet.with_length(data_tuples, 40)
    for paren in ("(", ")"):
        tok = tokenizer.t_to_i[paren]
        attn_probs_mean = avg_attn_probs_0_0(model, data_len_40, tokenizer, tok).detach().clone()
        px.imshow(
            attn_probs_mean,
            color_continuous_scale="RdBu", range_color=[0, 0.23], height=500, width=550,
            labels={"x": "Key Position", "y": "Query Position", "color": "Attn prob"},
            title=f"Attention patterns with query = {paren}", origin="lower"
        ).update_layout(margin=dict(l=60, r=60, t=80, b=40)).show()
# %%
if MAIN:
    tok = tokenizer.t_to_i["("]
    attn_probs_mean = avg_attn_probs_0_0(model, data_len_40, tokenizer, tok).detach().clone()
    px.bar(
        attn_probs_mean[1], 
        title=f"Attention pattern for first query position, query token = {paren!r}",
        labels={"index": "Sequence position", "value": "Average attention"}, 
        template="simple_white", 
        height=500, 
        width=600,
    ).update_layout(showlegend=False, margin_l=100, yaxis_range=[0, 0.1], hovermode="x unified").show()
# %%
def embedding_OV_0_0(model, emb_in: t.Tensor) -> t.Tensor:
    '''
    Takes an embedding such as open_paren and returns the output of 
    the 0.0 OV circuit when fed this embedding.
    '''
    attn = model.layers[0].self_attn
    ov = attn.W_O(attn.W_V(emb_in))
    ov = rearrange(
        ov,
        'b (n h) -> b n h',
        n=attn.num_heads,
    )
    return ov[:, 0, :].squeeze()

if MAIN:
    '''
    Fit the matrix L with the get_ln_fit function you wrote earlier.
    Combine the function embedding_OV_0_0 with the embedding function you wrote earlier to 
    calculate the vectors W_{OV}L(open_paren), W_{OV}L(close_paren).
    Verify that these two vectors are approximately opposite in direction and 
    equal in magnitude (for the former, you can use torch.cosine_similarity).
    '''
    norm1_ln_fit = get_ln_fit(model, data, model.layers[0].norm1, seq_pos=0)
    open_emb = embedding(model, tokenizer, '(')
    close_emb = embedding(model, tokenizer, ')')
    open_out = embedding_OV_0_0(model, open_emb)
    close_out = embedding_OV_0_0(model, close_emb)
    similarity = t.cosine_similarity(open_out, close_out, dim=-1).detach().numpy()
    ratio = (open_out.norm() / close_out.norm()).detach().numpy()
    assert similarity < -0.9
    assert .7 < ratio < 1.3
    print(similarity, ratio)
#%% [markdown]
#### Part 4: adversarial examples
#%%
# %%
if MAIN:
    # can leverage the red area of the attention grid for q~26, k~38
    examples = [
        "(((((((((((((((((((()))))))))))))))))))))", # balanced
        "())(()()()()()()()()()()()()()()()()()()", # A=2, B=36 # false "balanced" label 60%
        "((((((((((((((())))))))))))))))((((())))", # false balanced 99.986%, A=30, B=8
        "()()()()()()()()()()()()()()())(()()()()", # A=30, B=8 in different style
    ]
    m = max(len(ex) for ex in examples)
    toks = tokenizer.tokenize(examples).to(DEVICE)
    out = model(toks)
    print("\n".join([
        f"{ex:{m}} -> {p:.4%} balanced confidence" 
        for (ex, p) in zip(examples, out.exp()[:, 1])
    ]))
# %%
