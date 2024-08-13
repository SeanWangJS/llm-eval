"""
To align with https://github.com/EleutherAI/lm-evaluation-harness, we use the same tokenization functions.
"""

from typing import Tuple

import torch
from transformers import AutoTokenizer

def tok_encode(tokenizer: AutoTokenizer, string: str, add_special_tokens = False) -> torch.Tensor:
    """
    Align with lm-evaluation-harness/lm_eval/api/model.py
    """
    special_tokens_kwargs = {
        "add_special_tokens": add_special_tokens
    }
    encoding = tokenizer.encode(string, **special_tokens_kwargs, return_tensors="pt", truncation=True, max_length=1024)
    return encoding

def encode_pair(tokenizer: AutoTokenizer, ctx: str, cont: str, add_special_tokens = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    From lm-evaluation-harness/lm_eval/api/model.py
    """    
    n_spaces = len(ctx) - len(ctx.rstrip())
    if n_spaces > 0:
        cont = ctx[-n_spaces:] + cont
        ctx = ctx[:-n_spaces]

    whole_enc = tok_encode(tokenizer, ctx + cont, add_special_tokens)
    ctx_enc = tok_encode(tokenizer, ctx, add_special_tokens)

    ctx_enc_len = ctx_enc.size(1)
    cont_enc = whole_enc[:, ctx_enc_len:]

    return ctx_enc, cont_enc