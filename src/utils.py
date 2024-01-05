import os
import inspect
import math
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Text, List, Optional, Iterator, TypeVar

import torch
from torch.utils.data import Sampler, Dataset
from accelerate import Accelerator
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainerCallback, 
    StoppingCriteria,
    StoppingCriteriaList,
)


class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if args.should_save:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            kwargs["model"].save_pretrained(checkpoint_path)

            if "pytorch_model.bin" in os.listdir(checkpoint_path):
                os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


def prepare_model_for_peft(model, peft_config, args):
    from trl.import_utils import is_peft_available

    if is_peft_available():
        from peft import get_peft_model, prepare_model_for_kbit_training

    if not is_peft_available() and peft_config is not None:
        raise ValueError(
            "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
        )
    elif is_peft_available() and peft_config is not None:
        if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
            _support_gc_kwargs = hasattr(
                args, "gradient_checkpointing_kwargs"
            ) and "gradient_checkpointing_kwargs" in list(
                inspect.signature(prepare_model_for_kbit_training).parameters
            )

            preprare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

            if _support_gc_kwargs:
                preprare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

            model = prepare_model_for_kbit_training(model, **preprare_model_kwargs)
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        model = get_peft_model(model, peft_config)
    # For models that use gradient_checkpoiting, we need to attach a hook that enables input
    # to explicitly have `requires_grad=True`, otherwise training will either silently
    # fail or completely fail.
    elif getattr(args, "gradient_checkpointing", False):
        # For backward compatibility with older versions of transformers
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    return model


@Accelerator().on_local_main_process
def print_local_main(text):
    print(text)


def disable_progress_bar_non_local_main():
    if not Accelerator().is_local_main_process:
        import datasets
        import transformers
        import warnings
        datasets.utils.logging.disable_progress_bar()
        transformers.utils.logging.disable_progress_bar()
        warnings.filterwarnings('ignore')


def param_sharding_enabled():
    from transformers.modeling_utils import is_deepspeed_zero3_enabled, is_fsdp_enabled
    return is_deepspeed_zero3_enabled() or is_fsdp_enabled()


def prepare_input(data):
    # adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L2626
    if isinstance(data, Mapping):
        return type(data)({k: prepare_input(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_input(v) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": Accelerator().device}
        # TODO: inference-time deepspeed?
        # if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
        #     # NLP models inputs are int/uint and those get adjusted to the right dtype of the
        #     # embedding. Other models such as wav2vec2's inputs are already float and thus
        #     # may need special handling to match the dtypes of the model
        #     kwargs.update({"dtype": Accelerator().state.deepspeed_plugin.hf_ds_config.dtype()})
        return data.to(**kwargs)
    return data


@dataclass
class ChatInput:
    instruction: Text
    response: Text


@dataclass
class StopOnStringCriteria(StoppingCriteria):
    start_length: int
    eos_string: Text
    tokenizer: PreTrainedTokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length:])
        return all(self.eos_string in decoded_generation for decoded_generation in decoded_generations) # Stop when ALL sequences hit the stopping critera


def batch_generate_decode(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    inputs: List[Text], 
    eos_string=None,
    generation_configs=None,
) -> List[Text]:

    # tokenized and move to device
    inputs_tokens = prepare_input(tokenizer(inputs, padding=True, return_tensors="pt"))
    start_length = inputs_tokens["input_ids"].size(1)

    if eos_string:
        stopping_criteria = StoppingCriteriaList([
            StopOnStringCriteria(
                start_length=start_length,
                eos_string=eos_string,
                tokenizer=tokenizer,
            )
        ])
    else:
        stopping_criteria = None

    outputs_tokens = model.generate(
        input_ids=inputs_tokens["input_ids"],
        attention_mask=inputs_tokens["attention_mask"],
        stopping_criteria=stopping_criteria, # optional
        **generation_configs, #optional
    )
    outputs = tokenizer.batch_decode(
        outputs_tokens[:, start_length:],
        skip_special_tokens=True,
    )
    if eos_string:
        outputs = [output[:output.find(eos_string)] for output in outputs]
    return outputs


T_co = TypeVar('T_co', covariant=True)
class AsyncContiguousDistributedSampler(Sampler):
    ""
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None) -> None:
        if num_replicas is None:
            num_replicas = Accelerator().num_processes
        if rank is None:
            rank = Accelerator().process_index
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

        self.num_total_samples = len(dataset)
        self.num_local_samples = math.ceil(self.num_total_samples / self.num_replicas)
        self.local_starting_idx = self.num_local_samples * self.rank
        # self.num_local_samples = len(dataset)

    def __len__(self) -> int:
        return len(list(iter(self)))

    def __iter__(self) -> Iterator[T_co]:
        return iter(list(range(len(self.dataset)))[self.local_starting_idx:self.local_starting_idx+self.num_local_samples])
