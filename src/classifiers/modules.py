from abc import ABC
from dataclasses import dataclass, field
from typing import List, Text, Dict, Optional

import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils import prepare_input, param_sharding_enabled, ChatInput
 

@dataclass
class ClassiferBase(ABC):
    classification_template: Optional[Text] = "\n\nHuman:\n{instruction}\n\nAssistant:\n{response}"

    def __post_init__(self):
        assert "{instruction}" in self.classification_template and "{response}" in self.classification_template 

    def apply_chat_template(self, chat: ChatInput):
        return self.classification_template.format(instruction=chat.instruction, response=chat.response)

    def predict(instructions: List[ChatInput]) -> List[float]:
        """return the prob of positive(safe/positive/non-toxic etc.)"""
        raise NotImplementedError


@dataclass
class ClassifierBeaverTails(ClassiferBase):
    model_name: Optional[Text] = "meta-llama/Llama-2-7b-hf"
    load_in_4bit: Optional[bool] = False
    use_flash_attention_2: Optional[bool] = True

    def __post_init__(self):
        super().__post_init__()
        self.id2label = {0: "unsafe", 1: "safe"}
        self.label2id = {"unsafe": 0, "safe": 1}

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            id2label=self.id2label,
            label2id=self.label2id,
            torch_dtype=torch.float16,
            load_in_4bit=self.load_in_4bit,
            use_flash_attention_2=self.use_flash_attention_2,
            **({"device_map": {"": Accelerator().local_process_index}} if not param_sharding_enabled() else {}),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

    @torch.no_grad()
    def predict(self, chats_batch: List[ChatInput]) -> List[float]:
        inputs_batch_templated = [self.apply_chat_template(input) for input in chats_batch]
        inputs = prepare_input(self.tokenizer(inputs_batch_templated, padding=True, return_tensors="pt"))
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        return outputs.logits.softmax(-1)[:, -1].cpu().tolist()


if __name__ == "__main__":
    # PYTHONPATH=. srun -p llm-safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=8 python3 src/classifiers/modules.py
    chats_batch = [
        ChatInput("What's your name?", "fuck fuck fuck fuck fuck"),
        ChatInput("What's your name?", "Sam"),
    ]

    classifier = ClassifierBeaverTails(model_name="output/classifier/beavertails/best_checkpoint")
    predictions = classifier.predict(chats_batch)
    print(predictions)

