# PYTHONPATH=. srun -p llm-safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=8 python3 back_translate.py --input_path "output/red-teaming/llama-2-7b-fewshot/seed/score.jsonl" --output_path "output/red-teaming/llama-2-7b-fewshot/iter1/instruction.jsonl"
# PYTHONPATH=. srun -p llm-safety --quotatype=reserved --gres=gpu:8 --cpus-per-task=64 accelerate launch --num_processes=8 back_translate.py --input_path "output/red-teaming/llama-2-7b-fewshot/seed/score.jsonl" --output_path "output/red-teaming/llama-2-7b-fewshot/iter1/instruction.jsonl"
from dataclasses import dataclass, field
from typing import Optional

import tyro
import torch
import numpy as np
from datasets import load_dataset, Dataset
from accelerate import Accelerator

from src.utils import batchify_apply, ChatInput, set_seeds, print_local_main
from src.back_translators import BackTranslatorsRegistry, BackTranslateInput, ChatInput


@dataclass
class ScriptArguments:
    input_path: str = field(metadata={"help": "e.g., ***/score.jsonl"})
    output_path: str = field(metadata={"help": "e.g., ***/instruction.jsonl"})
    back_translator_model_cls: Optional[str] = field(default="llama-2-fewshot", metadata={"help": "e.g., `llama-2-fewshot`"})
    back_translator_model_name: Optional[str] = field(default=None, metadata={"help": "e.g., '/mnt/petrelfs/share_data/llm_llama/llama2/llama-2-7b-hf'"}) # 13b,70b
    seed: Optional[int] = field(default=1, metadata={"help": "seed"})
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "True or False"})
    examplars_sampling_temperature: Optional[float] = field(default=0.1)
    num_examplars: Optional[int] = field(default=1, metadata={"help": "the number of examplars for in-context learning"})
    num_new_instruction: Optional[int] = field(default=1000, metadata={"help": "number of new instructions to sample"})
script_args = tyro.cli(ScriptArguments)
print_local_main(script_args)
set_seeds(script_args.seed) # this is important

seed_results = load_dataset("json", data_files=script_args.input_path, split='train')
safety_probs = torch.Tensor(seed_results['safety_score'])
sample_probs = ((1-safety_probs) / script_args.examplars_sampling_temperature).softmax(-1)
sample_probs = sample_probs.numpy()
choices = np.random.choice(sample_probs.size, script_args.num_new_instruction, p=sample_probs)

def pop(dic: dict, key):
    dic.pop(key)
    return dic

inputs_dataset = [
    BackTranslateInput(
        # randomly choose examplars
        chat_examplars=[ChatInput(**pop(seed_results[np.random.choice(sample_probs.size)], key='safety_score')) for _ in range(script_args.num_examplars)],
        target_response=seed_results["response"][choice],
    )
    for choice in choices
]

if script_args.back_translator_model_name is None:
    bt = BackTranslatorsRegistry[script_args.back_translator_model_cls]()
else: 
    bt = BackTranslatorsRegistry[script_args.back_translator_model_cls](model_name=script_args.back_translator_model_name)

instructions = batchify_apply(
    bt.back_translate, 
    inputs_dataset=inputs_dataset, 
    per_device_batch_size=8, 
    split_among_ranks=True,
)

dataset = Dataset.from_dict({
    'instruction': instructions, 
    'instruction_before_translation': [seed_results['instruction'][choice] for choice in choices], 
    'response_before_translation': [seed_results['response'][choice] for choice in choices],
})

if Accelerator().is_main_process:
    dataset.to_json(script_args.output_path)
