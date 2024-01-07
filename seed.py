# python seed.py
from dataclasses import dataclass, field
from typing import Optional

import tyro
from datasets import load_dataset

@dataclass
class ScriptArguments:
    num_seed_instructions: Optional[int] = field(default=1000, metadata={"help": "number of seeding instructions"})
    output_path: Optional[str] = field(default="***/instruction.jsonl", metadata={"help": "the output path"})
script_args = tyro.cli(ScriptArguments)

dataset = load_dataset('PKU-Alignment/BeaverTails', split='30k_train')
dataset = dataset.select(range(script_args.num_seed_instructions))
dataset = dataset.select_columns('prompt').rename_column('prompt', 'instruction')
dataset.to_json(script_args.output_path)
