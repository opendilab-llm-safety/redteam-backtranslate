from abc import ABC, abstractclassmethod
from dataclasses import dataclass, field
from typing import List, Text, Optional

import jinja2
import torch
from transformers import StoppingCriteriaList
from accelerate import Accelerator

from src.utils import prepare_input, StopOnStringCriteria


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

    def _apply_instruction_augment_template(self, input: InstructionAugmentorInput) -> Text:
        return jinja2.Template(self.instruct_template).render(instruction_examplars=input)

    @abstractclassmethod
    def instruction_augment(self, inputs_batch: List[InstructionAugmentorInput]) -> List[Text]:
        raise NotImplementedError


@dataclass
class InstructionAugmentorLlama2(InstructionAugmentorBase):
    model_name: Text = "meta-llama/Llama-2-7b-hf"
    load_in_4bit: Optional[bool] = False
    use_flash_attention_2: Optional[bool] = True
    generation_configs: Optional[dict] = field(default_factory=lambda: {"do_sample":True, "max_length":512})

    def __post_init__(self):
        from transformers import LlamaForCausalLM, LlamaTokenizer
        self.model = LlamaForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            load_in_4bit=self.load_in_4bit,
            device_map={"": Accelerator().local_process_index},
            use_flash_attention_2=self.use_flash_attention_2,
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.stopping_criteria_fn = lambda start_length: StoppingCriteriaList([
            StopOnStringCriteria(
                start_length=start_length,
                eos_string=self.eos_string,
                tokenizer=self.tokenizer,
            )
        ])

    def instruction_augment(self, inputs_batch: List[InstructionAugmentorInput]) -> List[Text]:
        inputs_batch_templated = [self._apply_instruction_augment_template(input) for input in inputs_batch] 
        inputs = prepare_input(self.tokenizer(inputs_batch_templated, padding=True, return_tensors="pt"))
        start_length = inputs["input_ids"].size(1)
        outputs_tokens = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            stopping_criteria=self.stopping_criteria_fn(start_length=start_length),
            **self.generation_configs,
        )
        instructions = self.tokenizer.batch_decode(
            outputs_tokens[:, start_length:],
            skip_special_tokens=True,
        )
        instructions = [
            target_instruction[:target_instruction.find(self.eos_string)] 
            for target_instruction in instructions
        ]
        return instructions



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


    