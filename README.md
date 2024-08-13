# LLM-EVAL

This repo is in purpose of demonstrating how to evaluate a large language model(LLM) with various datasets. For those who are seeking for a framework which support widely models and datasets, please refer to [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). 

## Prequsites

We use [transformers](https://github.com/huggingface/transformers) as inference engine and [datasets](https://github.com/huggingface/datasets) to load the data.

## Usage

```shell
python -m llm_eval.hellaswag --model_name_or_path gpt2
```

## Result

|Model|Hellaswag|GPQA/gpqa_extended|
|---|---|---|
|GPT2|0.3114|0.2723|
|GPT2-gptq-4bit-128g| 0.3135| 
|google/Gemma-2b|0.7140|0.2912
|Qwen/Qwen2-7B-Instruct|0.8066|
|Qwen/Qwen2-7B-Instruct-gptq-4bit-128g | 0.7956|
|mistralai/Mistral-7B-Instruct-v0.3|0.7865|
|THUDM/glm-4-9b-chat|0.8063|
|THUDM/chatglm3-6b|0.6498|