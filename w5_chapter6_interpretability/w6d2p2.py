#%%
import circuitsvis as cv
cv.examples.hello("Bob")
# %%
from IPython import get_ipython
ipython = get_ipython()
# Code to automatically update the HookedTransformer code as its edited without restarting the kernel
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")
# %%
# import plotly.io as pio
# pio.renderers.default = "notebook_connected" # or use "browser" if you want plots to open with browser

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from einops import repeat, rearrange, reduce
from fancy_einsum import einsum
import random
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from torch.utils.data import DataLoader

from torchtyping import TensorType as TT
from torchtyping import patch_typeguard
from typeguard import typechecked
from typing import List, Union, Optional, Tuple
from functools import partial
from tqdm import tqdm
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML, display

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookedRootModule, HookPoint  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

def imshow(tensor, renderer=None, xaxis="", yaxis="", caxis="", **kwargs) -> go.Figure:
    fig = px.imshow(
        utils.to_numpy(tensor), 
        color_continuous_midpoint=0.0, 
        color_continuous_scale="RdBu", 
        labels={"x":xaxis, "y":yaxis, "color":caxis}, 
        **kwargs
    )
    fig.show(renderer)
    return fig

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs) -> go.Figure:
    fig = px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs)
    fig.show(renderer)
    return fig

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs) -> go.Figure:
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    fig = px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs)
    fig.show(renderer)
    return fig

#%%
device = 'cpu' # "cuda" if t.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

# %% [markdown]
#### Part 2: TransformerLens Features
# %%
example_text = "The first thing you need to figure out is *how* things are tokenized. `model.to_str_tokens` splits a string into the tokens *as a list of substrings*, and so lets you explore what the text looks like. To demonstrate this, let's use it on this paragraph."
example_text_str_tokens = model.to_str_tokens(example_text)
print(example_text_str_tokens)
#%%
example_text_tokens = model.to_tokens(example_text)
print(example_text_tokens)
# %%
example_multi_text = ["The cat sat on the mat.", "The cat sat on the mat really hard."]
example_multi_text_tokens = model.to_tokens(example_multi_text)
print(example_multi_text_tokens)
# %%
cat_text = "The cat sat on the mat."
cat_logits = model(cat_text) # shape [b, s, v] = [1, 8, 50257]
cat_probs = cat_logits.softmax(dim=-1)
print(f"Probability tensor shape [batch, position, d_vocab] == {cat_probs.shape}")

capital_the_token_index = model.to_single_token(" The")
print(f"| The| probability: {cat_probs[0, -1, capital_the_token_index].item():.2%}")
# %%
print(f"Token 256 - the most common pair of ASCII characters: |{model.to_string(256)}|")
# Squeeze means to remove dimensions of length 1. 
# Here, that removes the dummy batch dimension so it's a rank 1 tensor and returns a string
# Rank 2 tensors map to a list of strings
print(f"De-Tokenizing the example tokens: {model.to_string(example_text_tokens.squeeze())}")
# %%
print("With BOS:", model.get_token_position(" cat", "The cat sat on the mat"))
print(
    "Without BOS:", 
    model.get_token_position(" cat", "The cat sat on the mat", prepend_bos=False)
)
# %%
print("First occurence", model.get_token_position(
    " cat", 
    "The cat sat on the mat. The mat sat on the cat.", 
    mode="first"))
print("Final occurence", model.get_token_position(
    " cat", 
    "The cat sat on the mat. The mat sat on the cat.", 
    mode="last"))
# Defaults to first
# %%
print(model.to_str_tokens("2342+2017=21445"))
print(model.to_str_tokens("1000+1000000=999999"))

# %%
print("Logits shape by default (with BOS)", model("Hello World").shape)
print("Logits shape with BOS", model("Hello World", prepend_bos=True).shape)
print("Logits shape without BOS - only 2 positions!", model("Hello World", prepend_bos=False).shape)
# %%
# larger is better (desired answer is Claire)
ioi_logits_with_bos = model("Claire and Mary went to the shops, then Mary gave a bottle of milk to", prepend_bos=True)
mary_logit_with_bos = ioi_logits_with_bos[0, -1, model.to_single_token(" Mary")].item()
claire_logit_with_bos = ioi_logits_with_bos[0, -1, model.to_single_token(" Claire")].item()
print(f"Logit difference with BOS: {(claire_logit_with_bos - mary_logit_with_bos):.3f}")

ioi_logits_without_bos = model("Claire and Mary went to the shops, then Mary gave a bottle of milk to", prepend_bos=False)
mary_logit_without_bos = ioi_logits_without_bos[0, -1, model.to_single_token(" Mary")].item()
claire_logit_without_bos = ioi_logits_without_bos[0, -1, model.to_single_token(" Claire")].item()
print(f"Logit difference without BOS: {(claire_logit_without_bos - mary_logit_without_bos):.3f}")
# %%
print(f"| Claire| -> {model.to_str_tokens(' Claire', prepend_bos=False)}")
print(f"|Claire| -> {model.to_str_tokens('Claire', prepend_bos=False)}")
# %%
A = t.randn(5, 2)
B = t.randn(2, 5)
AB = A @ B
AB_factor = FactoredMatrix(A, B)
print("Norms:")
print(AB.norm())
print(AB_factor.norm())

print(f"Right dimension: {AB_factor.rdim}, Left dimension: {AB_factor.ldim}, Hidden dimension: {AB_factor.mdim}")
# %%
print("Eigenvalues:")
print(t.linalg.eig(AB).eigenvalues)
print(AB_factor.eigenvalues)
print()
print("Singular Values:")
print(t.linalg.svd(AB).S)
print(AB_factor.S)
# %%
C = t.randn(5, 300)
ABC = AB @ C
ABC_factor = AB_factor @ C
print("Unfactored:", ABC.shape, ABC.norm())
print("Factored:", ABC_factor.shape, ABC_factor.norm())
print(f"Right dimension: {ABC_factor.rdim}, Left dimension: {ABC_factor.ldim}, Hidden dimension: {ABC_factor.mdim}")
# %%
AB_unfactored = AB_factor.AB
print(t.isclose(AB_unfactored, AB).all())
# %%
OV_circuit_all_heads = model.OV
print(OV_circuit_all_heads)

OV_circuit_all_heads_eigenvalues = OV_circuit_all_heads.eigenvalues 
print(OV_circuit_all_heads_eigenvalues.shape)
print(OV_circuit_all_heads_eigenvalues.dtype)

OV_copying_score = (
    OV_circuit_all_heads_eigenvalues.sum(dim=-1).real / 
    OV_circuit_all_heads_eigenvalues.abs().sum(dim=-1)
)
imshow(
    utils.to_numpy(OV_copying_score), 
    xaxis="Head", 
    yaxis="Layer", 
    title="OV Copying Score for each head in GPT-2 Small", 
    zmax=1.0, 
    zmin=-1.0
)
# %%
scatter(
    x=OV_circuit_all_heads_eigenvalues[-1, -1, :].real, 
    y=OV_circuit_all_heads_eigenvalues[-1, -1, :].imag, 
    title="Eigenvalues of Head L11H11 of GPT-2 Small", 
    xaxis="Real", 
    yaxis="Imaginary"
)
# %%
full_OV_circuit = model.embed.W_E @ OV_circuit_all_heads @ model.unembed.W_U
print(full_OV_circuit)

full_OV_circuit_eigenvalues = full_OV_circuit.eigenvalues
print(full_OV_circuit_eigenvalues.shape)
print(full_OV_circuit_eigenvalues.dtype)

full_OV_copying_score = (
    full_OV_circuit_eigenvalues.sum(dim=-1).real / 
    full_OV_circuit_eigenvalues.abs().sum(dim=-1)
)
imshow(
    utils.to_numpy(full_OV_copying_score), 
    xaxis="Head", yaxis="Layer", 
    title="OV Copying Score for each head in GPT-2 Small", 
    zmax=1.0, 
    zmin=-1.0
)
# %%
scatter(
    x=full_OV_copying_score.flatten(), 
    y=OV_copying_score.flatten(), 
    hover_name=[f"L{layer}H{head}" for layer in range(12) for head in range(12)], 
    title="OV Copying Score for each head in GPT-2 Small", 
    xaxis="Full OV Copying Score", 
    yaxis="OV Copying Score"
)
# %%
model.generate(
    "(CNN) President Barack Obama caught in embarrassing new scandal\n", 
    max_new_tokens=50, 
    temperature=0.7, 
    prepend_bos=True
)
# %%
from transformer_lens.hook_points import HookedRootModule, HookPoint

class SquareThenAdd(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = nn.Parameter(t.tensor(offset))
        self.hook_square = HookPoint()

    def forward(self, x):
        # The hook_square doesn't change the value, but lets us access it
        square = self.hook_square(x * x)
        return self.offset + square

class TwoLayerModel(HookedRootModule):
    def __init__(self):
        super().__init__()
        self.layer1 = SquareThenAdd(3.0)
        self.layer2 = SquareThenAdd(-4.0)
        self.hook_in = HookPoint()
        self.hook_mid = HookPoint()
        self.hook_out = HookPoint()

        # We need to call the setup function of HookedRootModule to build an
        # internal dictionary of modules and hooks, and to give each hook a name
        super().setup()

    def forward(self, x):
        # We wrap the input and each layer's output in a hook - they leave the
        # value unchanged (unless there's a hook added to explicitly change it),
        # but allow us to access it.
        x_in = self.hook_in(x)
        x_mid = self.hook_mid(self.layer1(x_in))
        x_out = self.hook_out(self.layer2(x_mid))
        return x_out

model = TwoLayerModel().to(device=device)
# %%
out, cache = model.run_with_cache(t.tensor(5.0))
print("Model output:", out.item())
for key in cache:
    print(f"Value cached at hook {key}", cache[key].item())
# %%
def set_to_zero_hook(tensor, hook):
    print(hook.name)
    return t.tensor(0.0)

print(
    "Output after intervening on layer2.hook_scaled",
    model.run_with_hooks(
        t.tensor(5.0), fwd_hooks=[("layer2.hook_square", set_to_zero_hook)]
    ).item(),
)
# %%
from transformer_lens.loading_from_pretrained import get_checkpoint_labels
for model_name in ["attn-only-2l", "solu-12l", "stanford-gpt2-small-a"]:
    checkpoint_labels, checkpoint_label_type = get_checkpoint_labels(model_name)
    line(
        checkpoint_labels, 
        xaxis="Checkpoint Index", 
        yaxis=f"Checkpoint Value ({checkpoint_label_type})", 
        title=f"Checkpoint Values for {model_name} (Log scale)", 
        log_y=True, 
        markers=True
    )
for model_name in ["solu-1l-pile", "solu-6l-pile"]:
    checkpoint_labels, checkpoint_label_type = get_checkpoint_labels(model_name)
    line(
        checkpoint_labels, 
        xaxis="Checkpoint Index", 
        yaxis=f"Checkpoint Value ({checkpoint_label_type})", 
        title=f"Checkpoint Values for {model_name} (Linear scale)", 
        log_y=False, 
        markers=True
    )
# favour checkpoint_index syntax
#%%
@t.inference_mode()
def induction_loss(
    model, tokenizer=None, batch_size=4, subseq_len=384, prepend_bos=True
):
    """
    Generates a batch of random sequences repeated twice, and measures model performance on the second half. Tests whether a model has induction heads.

    By default, prepends a beginning of string token (prepend_bos flag), which is useful to give models a resting position, and sometimes models were trained with this.
    """
    # Make the repeated sequence
    first_half_tokens = t.randint(100, 20000, (batch_size, subseq_len)).to(device=device)
    repeated_tokens = repeat(first_half_tokens, "b p -> b (2 p)")

    # Prepend a Beginning Of String token
    if prepend_bos:
        if tokenizer is None:
            tokenizer = model.tokenizer
        repeated_tokens[:, 0] = tokenizer.bos_token_id
    # Run the model, and extract the per token correct log prob
    logits = model(repeated_tokens, return_type="logits")
    correct_log_probs = utils.lm_cross_entropy_loss(
        logits, repeated_tokens, per_token=True
    )
    # Take the loss over the second half of the sequence
    return correct_log_probs[:, subseq_len + 1 :].mean()
# %%
# from transformer_lens import evals
# We use the two layer model with SoLU activations, chosen fairly arbitrarily as 
# being both small (so fast to download and keep in memory) and 
# pretty good at the induction task.
model_name = "solu-2l"
# We can load a model from a checkpoint by specifying the checkpoint_index, 
# -1 means the final checkpoint
checkpoint_indices = [10, 25, 35, 60, -1]
checkpointed_models = []
tokens_trained_on = []
induction_losses = []
# %%
for index in checkpoint_indices:
    print(index)
    # Load the model from the relevant checkpoint by index
    model_for_this_checkpoint = HookedTransformer.from_pretrained(
        model_name, checkpoint_index=index, device=device,
    )
    checkpointed_models.append(model_for_this_checkpoint)

    tokens_seen_for_this_checkpoint = (
        model_for_this_checkpoint.cfg.checkpoint_value
    )
    tokens_trained_on.append(tokens_seen_for_this_checkpoint)

    induction_loss_for_this_checkpoint = induction_loss(
        model_for_this_checkpoint
    ).item()
    induction_losses.append(induction_loss_for_this_checkpoint)
# %%
fig = px.line(
    y=induction_losses, 
    x=tokens_trained_on, 
    labels={"x":"Tokens Trained On", "y":"Induction Loss"},
    title="Induction Loss over training: solu-2l", 
    markers=True, 
    log_x=True
) 
fig.show()
#%%
