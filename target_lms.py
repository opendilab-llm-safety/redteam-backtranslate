from abc import ABC
from dataclasses import dataclass, field
from typing import List, Text, Optional

import torch
from accelerate import Accelerator

from utils import prepare_input


class TargetBase(ABC):
    def get_responses(instructions: List[Text]) -> List[Text]:
        raise NotImplementedError


@dataclass
class TargetLlama2(TargetBase):
    use_flash_attention_2: Optional[bool] = True
    generation_configs: Optional[dict] = field(default_factory=lambda: {"do_sample":True, "max_length":512})

    def __post_init__(self):
        from transformers import LlamaForCausalLM, LlamaTokenizer
        self.model = LlamaForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map={"": Accelerator().local_process_index},
            use_flash_attention_2=self.use_flash_attention_2,
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
        self.tokenizer.add_bos_token = False # bos_token are already in chat template by default
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token


@dataclass
class TargetLlama2Chat(TargetLlama2):
    model_name: Optional[Text] = "meta-llama/Llama-2-7b-chat-hf"

    def get_responses(self, instructions: List[Text]) -> List[Text]:
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}], 
                tokenize=False
            ) 
            for instruction in instructions
        ]
        inputs = prepare_input(self.tokenizer(prompts, padding=True, return_tensors="pt"))

        outputs_tokens = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **self.generation_configs,
        )

        responses = [self.tokenizer.decode(output_tokens[len(input_tokens):], skip_special_tokens=True) 
            for input_tokens, output_tokens in zip(inputs["input_ids"], outputs_tokens)]
        breakpoint()
        
        return responses


"""
@dataclass
class TargetLlama2ZS(ABC):
    # zero-shot llama2
    temperature: Optional[float] = 0.6

    def __post_init__(self):
        pass

@dataclass
class TargetLlama2FS(ABC):
    # few-shot llama2
    temperature: Optional[float] = 0.6

    def __post_init__(self):
        pass
"""


if __name__ == '__main__':
    instructions = [
        "There's a llama in my garden 😱 What should I do?",
        "Who are you?"
    ]
    print("===Unit test for TargetLlama2Chat===")
    target_lm = TargetLlama2Chat(model_name="/mnt/petrelfs/share_data/llm_llama/llama2/llama-2-7b-chat-hf")
    responses = target_lm.get_responses(instructions)
    print("===[passed]===")
    breakpoint()