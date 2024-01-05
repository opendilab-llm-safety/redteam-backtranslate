from abc import ABC, abstractclassmethod
from dataclasses import dataclass, field
from typing import List, Text, Optional

import jinja2
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

from src.utils import batch_generate_decode


InstructionAugmentorInput = List[Text]


@dataclass
class InstructionAugmentorBase:
    instruct_template: Optional[Text] = (
"""List of questions to ask someone:
{% for instruction_examplar in instruction_examplars %}
{{ loop.index }}: {{ instruction_examplar }}
{% endfor %}
{{ instruction_examplars | length + 1}}:
""")
    eos_string: Optional[Text] = "\n\n"

    def __post_init__(self):
        pass # TODO: assert keywords in template

    def _apply_instruction_augment_template(self, input: InstructionAugmentorInput) -> Text:
        return jinja2.Template(self.instruct_template).render(instruction_examplars=input)

    @abstractclassmethod
    def instruction_augment(self, inputs_batch: List[InstructionAugmentorInput]) -> List[Text]:
        raise NotImplementedError


@dataclass
class InstructionAugmentorHFBase(InstructionAugmentorBase):
    generation_configs: Optional[dict] = field(default_factory=lambda: {"do_sample":True, "max_new_tokens":512})

    def __post_init__(self):
        super().__post_init__()
        self.model, self.tokenizer = self._init_model_and_tokenizer()

    @abstractclassmethod
    def _init_model_and_tokenizer(self):
        raise NotImplementedError

    def instruction_augment(self, inputs_batch: List[InstructionAugmentorInput]) -> List[Text]:
        inputs_batch_templated = [self._apply_instruction_augment_template(input) for input in inputs_batch]
        return batch_generate_decode(
            model=self.model,
            tokenizer=self.tokenizer, 
            inputs=inputs_batch_templated, 
            eos_string=self.eos_string, 
            generation_configs=self.generation_configs,
        )


@dataclass
class InstructionAugmentorLlama2(InstructionAugmentorHFBase):
    model_name: Text = "meta-llama/Llama-2-7b-hf"
    load_in_4bit: Optional[bool] = False
    use_flash_attention_2: Optional[bool] = True

    def _init_model_and_tokenizer(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            load_in_4bit=self.load_in_4bit,
            device_map={"": Accelerator().local_process_index},
            use_flash_attention_2=self.use_flash_attention_2,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        return model, tokenizer


if __name__ == "__main__":
    # PYTHONPATH=. srun -p llm-safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=8 python3 src/instruction_augmentors.py
    inputs_batch = [
        [
            "What is the worst piece of advice you’ve ever received?",
            "If you could go back in time, what would you do?",
        ]
    ] * 2
    template = InstructionAugmentorBase()._apply_instruction_augment_template(inputs_batch[0])
    print(template)
    print("\n\n")

    print("=== InstructionAugmentorLlama2 ===")
    bt = InstructionAugmentorLlama2(model_name="/mnt/petrelfs/share_data/llm_llama/llama2/llama-2-7b-hf")
    instructions = bt.instruction_augment(inputs_batch)
    for instruction in instructions:
        print(instruction)
    print("=== [passed] ===")
    breakpoint()


    