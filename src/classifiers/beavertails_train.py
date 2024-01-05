import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import evaluate
import tyro
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig
from accelerate import Accelerator
from datasets import load_dataset 

from src.utils import print_local_main, disable_progress_bar_non_local_main, prepare_model_for_peft, PeftSavingCallback, ChatInput
from src.classifiers.modules import ClassifierBeaverTails

disable_progress_bar_non_local_main()

@dataclass
class ScriptArguments:

    model_name: str = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the base model name"})
    use_flash_attention_2: Optional[bool] = field(default=True, metadata={"help": "whether to use flash attention 2"})
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "whether to conduct sanity check"})
    num_proc: Optional[int] = field(default=4, metadata={"help": "num_proc for dataset.map"})

    training_args: TrainingArguments = field(
        default_factory=lambda: TrainingArguments(
            output_dir="./output/classifier/beavertails",
            overwrite_output_dir=True,

            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            warmup_steps=0.1,
            weight_decay=0.05,
            bf16=True,
            remove_unused_columns=False,
            run_name="classifier/beavertails",
            report_to="wandb",

            num_train_epochs=2,
            logging_steps=10,
            save_steps=0.2,
            eval_steps=0.2,
            eval_delay=0.2,
            evaluation_strategy="steps",
            save_total_limit=2,
            metric_for_best_model="accuracy",
            load_best_model_at_end=True,
        )
    )

    peft: Optional[bool] = field(default=False, metadata={"help": "whether to use peft for training"})
    peft_config: LoraConfig = field(
        default_factory=lambda: LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
            modules_to_save=["score"], # maybe optional
        )
    )
script_args = tyro.cli(ScriptArguments)

# base model
print_local_main("loading model...")
classifier = ClassifierBeaverTails(
    model_name=script_args.model_name, 
    use_flash_attention_2=script_args.use_flash_attention_2
)
model, tokenizer = classifier.model, classifier.tokenizer
print_local_main(model)

if script_args.peft:
    print_local_main("initializing peft...")
    model = prepare_model_for_peft(model, script_args.peft_config, script_args.training_args)
    if Accelerator().is_local_main_process: model.print_trainable_parameters()
    callbacks = [PeftSavingCallback]
else:
    callbacks = []

def map_func(examples):
    instructions = examples["prompt"]
    responses = examples["response"]
    chats = [classifier.apply_chat_template(ChatInput(instruction, response)) for instruction, response in zip(instructions, responses)]
    labels = [int(label) for label in examples["is_safe"]]
    return {"label":labels, **tokenizer(chats, truncation=False)}

def preprocess_dataset(dataset):
    if script_args.sanity_check:
        dataset = dataset.select(range(min(len(dataset), 100)))
    return dataset.map(map_func, batched=True, num_proc=script_args.num_proc, remove_columns=dataset.column_names)

# dataset 
train_dataset = preprocess_dataset(load_dataset('PKU-Alignment/BeaverTails', split='330k_train'))
eval_dataset  = preprocess_dataset(load_dataset('PKU-Alignment/BeaverTails', split='330k_test'))

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=script_args.training_args,
    data_collator=DataCollatorWithPadding(tokenizer),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=callbacks,
    compute_metrics=compute_metrics,
)
trainer.train()

save_name = "best_checkpoint" if script_args.training_args.load_best_model_at_end else "final_checkpoint"
trainer.model.save_pretrained(os.path.join(script_args.training_args.output_dir, save_name))
trainer.tokenizer.save_pretrained(os.path.join(script_args.training_args.output_dir, save_name))
