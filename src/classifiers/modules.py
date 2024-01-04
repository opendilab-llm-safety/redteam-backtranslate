from abc import ABC
from dataclasses import dataclass, field
from typing import List, Text, Dict, Optional

import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils import prepare_input, param_sharding_enabled
 

@dataclass
class ChatInput:
    instruction: Text
    response: Text


class ClassiferBase(ABC):
    def get_predictions(instructions: List[Text]) -> List[Text]:
        raise NotImplementedError


@dataclass
class ClassifierBeaverTails(ClassiferBase):
    model_name: Optional[Text] = "meta-llama/Llama-2-7b-hf"
    use_flash_attention_2: Optional[bool] = True
    classification_template: Optional[Text] = "\n\nHuman:\n{instruction}\n\nAssistant:\n{response}"

    def __post_init__(self):
        self.id2label = {0: "unsafe", 1: "safe"}
        self.label2id = {"unsafe": 0, "safe": 1}
        assert "{instruction}" in self.classification_template and "{response}" in self.classification_template 

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            id2label=self.id2label,
            label2id=self.label2id,
            use_flash_attention_2=self.use_flash_attention_2,
            **({"device_map": {"": Accelerator().local_process_index}} if not param_sharding_enabled() else {}),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def apply_chat_template(self, chat: ChatInput):
        return self.classification_template.format(instruction=chat.instruction, response=chat.response)

    def get_predictions(instructions: List[ChatInput]) -> List[Text]:
        pass
