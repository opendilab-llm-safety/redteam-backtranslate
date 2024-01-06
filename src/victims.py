from abc import ABC
from dataclasses import dataclass, field
from typing import List, Text, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

from src.utils import batch_generate_decode


class VictimBase(ABC):
    def respond(instructions: List[Text]) -> List[Text]:
        raise NotImplementedError


@dataclass
class VictimLlama2(VictimBase, ABC):
    model_name: Optional[Text] = "meta-llama/Llama-2-7b-hf"
    load_in_4bit: Optional[bool] = False
    use_flash_attention_2: Optional[bool] = True
    generation_configs: Optional[dict] = field(default_factory=lambda: {"do_sample":True, "max_new_tokens":512})

    def __post_init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            load_in_4bit=self.load_in_4bit,
            device_map={"": Accelerator().local_process_index},
            use_flash_attention_2=self.use_flash_attention_2,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id


@dataclass
class VictimLlama2Chat(VictimLlama2):
    model_name: Optional[Text] = "meta-llama/Llama-2-7b-chat-hf"

    def __post_init__(self):
        super().__post_init__()
        self.tokenizer.add_bos_token = False # bos_token are already in chat template by default

    def respond(self, instructions: List[Text]) -> List[Text]:
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}], 
                tokenize=False
            ) 
            for instruction in instructions
        ]
        
        return batch_generate_decode(
            model=self.model,
            tokenizer=self.tokenizer, 
            inputs=prompts, 
            generation_configs=self.generation_configs,
        )


if __name__ == '__main__':
    # PYTHONPATH=. srun -p llm-safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=8 python3 src/victims.py
    instructions = [
        "There's a llama in my garden 😱 What should I do?",
        "Who are you?"
    ]
    generation_configs = {"do_sample":False, "max_new_tokens":512}

    print("=== VictimLlama2Chat ===")
    target_lm = VictimLlama2Chat(
        model_name="/mnt/petrelfs/share_data/llm_llama/llama2/llama-2-7b-chat-hf", 
        generation_configs=generation_configs
    )
    responses = target_lm.respond(instructions)
    for response in responses: print(response)
    print("=== [passed] ===")

    print("=== VictimLlama2Chat(4bit) ===")
    target_lm = VictimLlama2Chat(
        model_name="/mnt/petrelfs/share_data/llm_llama/llama2/llama-2-7b-chat-hf", 
        load_in_4bit=True, 
        generation_configs=generation_configs
    )
    responses = target_lm.respond(instructions)
    for response in responses: print(response)
    print("=== [passed] ===")
    breakpoint()
