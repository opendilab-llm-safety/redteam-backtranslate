# PYTHONPATH=. srun -p llm-safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=8 python3 respond.py
# PYTHONPATH=. srun -p llm-safety --quotatype=reserved --gres=gpu:8 --cpus-per-task=64 accelerate launch --num_processes=8 respond.py
from dataclasses import dataclass, field
from typing import Optional

import tyro
from datasets import load_dataset, Dataset
from accelerate import Accelerator

from src.utils import batchify_apply, set_seeds
from src.victims.victims import VictimsRegistry


@dataclass
class ScriptArguments:
    input_path: str = field(metadata={"help": "e.g., ***/instruction.jsonl"})
    output_path: str = field(metadata={"help": "e.g., ***/instruction_response.jsonl"})
    victim_model_cls: str = field(default="llama-2-fewshot", metadata={"help": "e.g., `llama-2-fewshot`"})
    victim_model_name: Optional[str] = field(default=None, metadata={"help": "e.g., '/mnt/petrelfs/share_data/llm_llama/llama2/llama-2-7b-hf'"}) # 13b,70b
    seed: Optional[int] = field(default=1, metadata={"help": "seed"})
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "True or False"})
    def __post_init__(self):
        assert self.victim_model_cls in VictimsRegistry
        # f"victim_model({self.victim_model_cls}) should be with in {'\n'.join(VictimsRegistry.keys())}" 
script_args = tyro.cli(ScriptArguments)
set_seeds(script_args.seed)

instructions_dataset = load_dataset("json", data_files=script_args.input_path, split='train')

if script_args.sanity_check:
    instructions_dataset = instructions_dataset.select(range(20))

target_lm = VictimsRegistry[script_args.victim_model_cls](model_name=script_args.victim_model_name)

responses = batchify_apply(
    target_lm.respond, 
    instructions_dataset['instruction'],
    per_device_batch_size=8, 
    split_among_ranks=True,
)

if Accelerator().is_main_process():
    dataset = Dataset.from_dict({'response': responses, 'instruction': instructions_dataset['instruction']})
    dataset.to_json(script_args.output_path)
