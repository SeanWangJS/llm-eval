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

def hellaswag_eval(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, docs: List[Dict[str, str]]):

    if model.config.architectures[0].startswith("Gemma"):
        add_special_tokens = True
    else:
        add_special_tokens = False

    correct = 0
    loop = tqdm(enumerate(docs), total=len(docs))
    for i, doc in loop:
        label = int(doc["label"])
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        context = preprocess(doc["activity_label"] + ": " + ctx)
        choices = [preprocess(ending) for ending in doc["endings"]]
        lls = []
        for continuation in choices:
            context_enc, continuation_enc = encode_pair(tokenizer, context, target_delimiter + continuation, add_special_tokens)
            context_enc = context_enc.to("cuda")
            continuation_enc = continuation_enc.to("cuda")
            continuation_len = continuation_enc.size(1)

            input_ids = torch.cat([context_enc, continuation_enc], dim=1)
            input_ids = input_ids[:, :-1]

            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits ## [1, seq_len, vocab_size]
            logits = logits[:, -continuation_len:, :]
            logits = F.log_softmax(logits, dim=-1)

            logits = torch.gather(logits, 2, continuation_enc.unsqueeze(-1)) ## [1, seq_len, 1]
            
            ## We only record the normalized log likelihood
            lls.append(logits.sum().item() / len(continuation))

        pred = np.argmax(lls)
        if pred == label:
            correct += 1

        acc = correct / (i + 1)
        loop.set_description(f"Accuracy: {acc:.6f}")
    
    print(f"Accuracy: {correct / len(docs)}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the huggingface model")
    parser.add_argument("--data_dir", type=str, required=False, help="Path to the Hellaswag dataset, if not provided, the dataset will be downloaded from Huggingface")
    args = parser.parse_args()

    model_name_or_path = args.model_name_or_path
    data_dir = args.data_dir

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map = "auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if data_dir:
        val_data_path = os.path.join(data_dir, "hellaswag_val.jsonl")
        with open(val_data_path, "r") as f:
            docs = [json.loads(line) for line in f]
    else:
        dataset = load_dataset("hellaswag")
        docs = dataset["validation"]

    hellaswag_eval(model, tokenizer, docs)