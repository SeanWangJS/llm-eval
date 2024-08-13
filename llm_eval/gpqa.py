import os
import argparse
from typing import List, Tuple, Dict
import json
import re
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from llm_eval.utils import encode_pair

def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

target_delimiter = " "

def gpqa_eval(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataset: List[Dict[str, str]]):

    if model.config.architectures[0].startswith("Gemma"):
        add_special_tokens = True
    else:
        add_special_tokens = False

    prompt_template = "What is the correct answer to this question:{Question}\nChoices:\n(A) {choice1}\n(B) {choice2}\n(C) {choice3}\n(D) {choice4}\nAnswer: "

    loop = tqdm(enumerate(dataset), total=len(dataset))
    num_correct = 0
    for i, sample in loop:

        choices = [
            preprocess(sample["Incorrect Answer 1"]),
            preprocess(sample["Incorrect Answer 2"]),
            preprocess(sample["Incorrect Answer 3"]),
            preprocess(sample["Correct Answer"])
        ]

        random.shuffle(choices)
        label = choices.index(preprocess(sample["Correct Answer"]))

        question = sample["Question"]

        ctx = prompt_template.format(
            Question=question,
            choice1=choices[0],
            choice2=choices[1],
            choice3=choices[2],
            choice4=choices[3]
        )

        lls = []
        for choice_id, _ in enumerate(choices):
            cont = target_delimiter + f"({chr(65 + choice_id)})"

            ctx_enc, cont_enc = encode_pair(tokenizer, ctx,  cont, add_special_tokens)
            ctx_enc = ctx_enc.to("cuda")
            cont_enc = cont_enc.to("cuda")
            cont_len = cont_enc.size(1)

            input_ids = torch.cat([ctx_enc, cont_enc], dim=1) ## [1, L]
            input_ids = input_ids[:, :-1]

            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits ## [1, L-1, vocab_size]
            logits = logits[:, -cont_len:, :]
            probs = F.log_softmax(logits, dim=-1) ## [1, cont_len, vocab_size]

            probs = torch.gather(logits, 2, cont_enc.unsqueeze(-1)) ## [1, seq_len, 1]

            lls.append(probs.sum().item() / len(cont))

        pred = np.argmax(lls)
        if pred == label:
            num_correct += 1

        acc = num_correct / (i + 1)
        loop.set_description(f"Accuracy: {acc:.6f}")

    acc = num_correct / len(dataset)
    print(f"Accuracy: {acc:.6f}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the huggingface model")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, 
                                                 device_map = "auto", 
                                                 torch_dtype = torch.float16
                                                 )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    dataset = load_dataset("gpqa", "gpqa_extended")
    dataset = dataset["train"]

    gpqa_eval(model, tokenizer, dataset)

    