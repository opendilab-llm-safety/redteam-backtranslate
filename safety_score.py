# PYTHONPATH=. srun -p llm-safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=8 python3 safety_score.py
# PYTHONPATH=. srun -p llm-safety --quotatype=reserved --gres=gpu:8 --cpus-per-task=64 accelerate launch --num_processes=8 safety_score.py
from dataclasses import dataclass, field
from typing import Optional

import tyro
from datasets import load_dataset, Dataset
from accelerate import Accelerator

from src.utils import batchify_apply, ChatInput, set_seeds
from src.classifiers.modules import ClassifierBeaverTails


@dataclass
class ScriptArguments:
    input_path: str = field(metadata={"help": "e.g., ***/instruction_response.jsonl"})
    output_path: str = field(metadata={"help": "e.g., ***/score.jsonl"})
    classifier_model_name: Optional[str] = field(default="output/classifier/beavertails/best_checkpoint", metadata={"help": "e.g., 'output/classifier/beavertails/best_checkpoint'"})
    seed: Optional[int] = field(default=1, metadata={"help": "seed"})
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "True or False"})
script_args = tyro.cli(ScriptArguments)
set_seeds(script_args.seed)

instructions_responses_dataset = load_dataset("json", data_files=script_args.input_path, split='train')

if script_args.sanity_check:
    instructions_responses_dataset = instructions_responses_dataset.select(range(20))

classifier = ClassifierBeaverTails(model_name=script_args.classifier_model_name)

prob = batchify_apply(
    classifier.predict, 
    inputs_dataset=[ChatInput(instruction, response) for instruction, response in zip(instructions_responses_dataset['instruction'], instructions_responses_dataset['response'])], 
    per_device_batch_size=8, 
    split_among_ranks=True,
)

if Accelerator().is_main_process:
    dataset = Dataset.from_dict({'safety_score': prob, 'response': instructions_responses_dataset['response'], 'instruction': instructions_responses_dataset['instruction']})
    dataset.to_json(script_args.output_path)
