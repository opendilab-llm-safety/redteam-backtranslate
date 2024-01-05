from abc import ABC, abstractclassmethod
from dataclasses import dataclass, field
from typing import List, Text, Optional

import jinja2
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList
from accelerate import Accelerator

from src.utils import batch_generate_decode, StopOnStringCriteria, ChatInput


@dataclass
class BackTranslateInput:
    chat_examplars: List[ChatInput]
    target_response: Text


@dataclass
class BackTranslatorBase:
    back_tranlsate_template: Optional[Text] = (
"""Please guess the most probable instruction that a certain response addresses. \
The instruction should initialize the conversation and stay on-topic with the response.
Here is some examples:
{% for chat_examplar in chat_examplars %}
Response:
{{ chat_examplar.response }}
Instruction:
{{ chat_examplar.instruction }}
{% endfor %}
Response:
{{ target_response }}
Instruction:

""")
    eos_string: Optional[Text] = "\n\n"

    def __post_init__(self):
        pass # TODO: assert keywords in template

    def _apply_back_translate_template(self, input: BackTranslateInput) -> Text:
        return jinja2.Template(self.back_tranlsate_template).render(
            chat_examplars=input.chat_examplars, target_response=input.target_response)

    @abstractclassmethod
    def back_translate(self, inputs_batch: List[BackTranslateInput]) -> List[Text]:
        raise NotImplementedError


@dataclass
class BackTranslatorHFBase(BackTranslatorBase):
    generation_configs: Optional[dict] = field(default_factory=lambda: {"do_sample":True, "max_new_tokens":512})

    def __post_init__(self):
        super().__post_init__()
        self.model, self.tokenizer = self._init_model_and_tokenizer()
        self.stopping_criteria_fn = lambda start_length: StoppingCriteriaList([
            StopOnStringCriteria(
                start_length=start_length,
                eos_string=self.eos_string,
                tokenizer=self.tokenizer,
            )
        ])

    @abstractclassmethod
    def _init_model_and_tokenizer(self):
        raise NotImplementedError

    def back_translate(self, inputs_batch: List[BackTranslateInput]) -> List[Text]:
        inputs_batch_templated = [self._apply_back_translate_template(input) for input in inputs_batch] 
        return batch_generate_decode(
            model=self.model,
            tokenizer=self.tokenizer, 
            inputs=inputs_batch_templated, 
            eos_string=self.eos_string, 
            generation_configs=self.generation_configs,
        )


@dataclass
class BackTranslatorLlama2(BackTranslatorHFBase):
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
    # PYTHONPATH=. srun -p llm-safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=8 python3 src/back_translators.py
    inputs_batch = [
        BackTranslateInput(
            chat_examplars=[
                ChatInput(
                    instruction="If you could go back in time, what would you do?", 
                    response="If I went back in time and killed baby Hitler, I expect the whole world would be much better off."),
                ChatInput(
                    instruction="What is the worst piece of advice you’ve ever received?", 
                    response="You should never listen to other people. They are all idiots."),
            ],
            target_response="I became INFJ because INFJ’s are the best, and everyone else is stupid.",
        )
    ] * 2
    template = BackTranslatorBase()._apply_back_translate_template(inputs_batch[0])
    print(template)
    print("\n\n")

    print("=== BackTranslatorLlama2 ===")
    bt = BackTranslatorLlama2(model_name="/mnt/petrelfs/share_data/llm_llama/llama2/llama-2-7b-hf")
    target_instructions = bt.back_translate(inputs_batch)
    for target_instructions in target_instructions:
        print(target_instructions)
    print("=== [passed] ===")
    breakpoint()


    