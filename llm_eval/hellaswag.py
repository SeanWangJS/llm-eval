import os
import argparse
from typing import List, Tuple, Dict
import json
import re

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from llm_eval.utils import encode_pair

## To align with lm-evaluation-harness, we set the target delimiter to " "
target_delimiter = " "

def preprocess(text):
    """
    From lm-evaluation-harness/lm_eval/tasks/hellaswag/utils.py
    """
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def hellaswag_eval(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, samples: List[Dict[str, str]]):

    if model.config.architectures[0].startswith("Gemma"):
        add_special_tokens = True
    else:
        add_special_tokens = False

    correct = 0
    loop = tqdm(enumerate(samples), total=len(samples))
    for i, sample in loop:
        label = int(sample["label"])
        ctx = sample["ctx_a"] + " " + sample["ctx_b"].capitalize()
        ctx = preprocess(sample["activity_label"] + ": " + ctx)
        choices = [preprocess(ending) for ending in sample["endings"]]
        lls = []
        for cont in choices:
            ctx_enc, cont_enc = encode_pair(tokenizer, ctx, target_delimiter + cont, add_special_tokens)
            ctx_enc = ctx_enc.to("cuda")
            cont_enc = cont_enc.to("cuda")
            cont_len = cont_enc.size(1)

            input_ids = torch.cat([ctx_enc, cont_enc], dim=1)
            input_ids = input_ids[:, :-1]

            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits ## [1, seq_len, vocab_size]
            logits = logits[:, -cont_len:, :]
            logits = F.log_softmax(logits, dim=-1)

            logits = torch.gather(logits, 2, cont_enc.unsqueeze(-1)) ## [1, seq_len, 1]
            
            ## We only record the normalized log likelihood
            lls.append(logits.sum().item() / len(cont))

        pred = np.argmax(lls)
        if pred == label:
            correct += 1

        acc = correct / (i + 1)
        loop.set_description(f"Accuracy: {acc:.6f}")
    
    print(f"Accuracy: {correct / len(samples)}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the huggingface model")
    args = parser.parse_args()

    model_name_or_path = args.model_name_or_path

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map = "auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    dataset = load_dataset("hellaswag")
    samples = dataset["validation"]

    hellaswag_eval(model, tokenizer, samples)