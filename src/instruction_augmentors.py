from abc import ABC, abstractclassmethod
from dataclasses import dataclass, field
from typing import List, Text, Optional

import jinja2
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel
from accelerate import Accelerator

from src.utils import batch_generate_decode, Registry


InstructionAugmentorsRegistry = Registry()


InstructionAugmentorInput = List[Text]


@dataclass
class InstructionAugmentorBase:
    instruct_template: Optional[Text] = (
"""List of questions to ask someone:
{% for instruction_examplar in instruction_examplars %}
{{ loop.index }}:```{{ instruction_examplar }}```
{% endfor %}
{{ instruction_examplars | length + 1 }}:```\
""")
    eos_string: Optional[Text] = "```\n\n"

    def __post_init__(self):
        pass # TODO: assert keywords in template

    def apply_instruction_augment_template(self, input: InstructionAugmentorInput) -> Text:
        return jinja2.Template(self.instruct_template).render(instruction_examplars=input)

    @abstractclassmethod
    def instruction_augment(self, inputs_batch: List[InstructionAugmentorInput]) -> List[Text]:
        raise NotImplementedError


@dataclass
class InstructionAugmentorHFBase(InstructionAugmentorBase):
    model: Optional[PreTrainedModel] = None
    model_name: Optional[Text] = None
    generation_configs: Optional[dict] = field(default_factory=lambda: {"do_sample":True, "max_new_tokens":512})

    def __post_init__(self):
        super().__post_init__()
        self.model, self.tokenizer = self._init_model_and_tokenizer()

    @abstractclassmethod
    def _init_model_and_tokenizer(self):
        raise NotImplementedError

    def instruction_augment(self, inputs_batch: List[InstructionAugmentorInput]) -> List[Text]:
        inputs_batch_templated = [self.apply_instruction_augment_template(input) for input in inputs_batch]
        return batch_generate_decode(
            model=self.model,
            tokenizer=self.tokenizer, 
            inputs=inputs_batch_templated, 
            eos_string=self.eos_string, 
            generation_configs=self.generation_configs,
        )


@InstructionAugmentorsRegistry.register("llama-2-fewshot")
@dataclass
class InstructionAugmentorLlama2FewShot(InstructionAugmentorHFBase):
    model_name: Text = "meta-llama/Llama-2-7b-hf"
    load_in_4bit: Optional[bool] = False
    use_flash_attention_2: Optional[bool] = True

    def _init_model_and_tokenizer(self):
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                load_in_4bit=self.load_in_4bit,
                device_map={"": Accelerator().local_process_index},
                use_flash_attention_2=self.use_flash_attention_2,
            )
        # make an extra copy of a tokenizer anyway
        self.tokenizer = AutoTokenizer.from_pretrained(self.model.config._name_or_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.model.config.pad_token_id = self.tokenizer.eos_token_id
        return self.model, self.tokenizer


@InstructionAugmentorsRegistry.register("llama-2-zeroshot")
@dataclass
class InstructionAugmentorLlama2ZeroShot(InstructionAugmentorLlama2FewShot):
    def apply_instruction_augment_template(self, input: InstructionAugmentorInput) -> Text:
        return super().apply_instruction_augment_template([])


if __name__ == "__main__":
    # PYTHONPATH=. srun -p llm-safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=8 python3 src/instruction_augmentors.py
    inputs_batch = [
        [
            "What is the worst piece of advice you’ve ever received?",
            "If you could go back in time, what would you do?",
        ]
    ] * 2
    # template = InstructionAugmentorBase().apply_instruction_augment_template(inputs_batch[0])
    # print(template)
    # print("\n\n")
    model_name="/mnt/petrelfs/share_data/llm_llama/llama2/llama-2-70b-hf"
    load_in_4bit=True
    generation_configs = {"do_sample":True, "max_new_tokens":128, "temperature":1.0}

    print("#"*40 + " [InstructionAugmentorLlama2FewShot] " + "#"*40)
    ia = InstructionAugmentorLlama2FewShot(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
        generation_configs=generation_configs
    )
    print('-'*40 + ' [Prompt Start] ' + '-'*40)
    print(ia.apply_instruction_augment_template(inputs_batch[0]))
    print('-'*40 + ' [Prompt End] ' + '-'*40)
    instructions = ia.instruction_augment(inputs_batch)
    for instruction in instructions:
        print(instruction)
    print("=== [passed] ===")
    breakpoint()
    