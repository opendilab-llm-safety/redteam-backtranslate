import os
import inspect
import math
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Text, List, Any, Callable, Optional, Iterator, TypeVar

import tqdm
import numpy as np
import torch
from torch.utils.data import Sampler, Dataset, DataLoader
import accelerate
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


def pad_labels(features, tokenizer, pad_to_multiple_of=None, label_pad_token_id=-100):
    # copied from https://github.com/huggingface/transformers/blob/v4.34.1/src/transformers/data/data_collator.py#L562-L584
    labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
    # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
    # same length to return tensors.
    if labels is not None:
        max_label_length = max(len(l) for l in labels)
        if pad_to_multiple_of is not None:
            max_label_length = (
                (max_label_length + pad_to_multiple_of - 1)
                // pad_to_multiple_of
                * pad_to_multiple_of
            )

        padding_side = tokenizer.padding_side
        for feature in features:
            remainder = [label_pad_token_id] * (max_label_length - len(feature["labels"]))
            if isinstance(feature["labels"], list):
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )
            elif padding_side == "right":
                feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
            else:
                feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)


def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
    label_pad_token_id: int = -100,
) -> torch.FloatTensor:
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == label_pad_token_id] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


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

    err_cnt = 0
    # catch `RuntimeError: probability tensor contains either `inf`, `nan` or element < 0`
    while True:
        try:
            outputs_tokens = model.generate(
                input_ids=inputs_tokens["input_ids"],
                attention_mask=inputs_tokens["attention_mask"],
                stopping_criteria=stopping_criteria,
                **generation_configs,
            )
            break
        except RuntimeError:
            err_cnt += 1
            if err_cnt >= 3: generation_configs["do_sample"] = False
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


def set_seeds(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def batchify_apply(
    fn: Callable,
    inputs_dataset: List[Any], 
    per_device_batch_size: int = 8,
    split_among_ranks: bool = True, # split data among ranks and then all gather 
) -> List[Any]:
    """
    This function splits `inputs_dataset` into multiple batchs, apply `fn` to each batch, and return the collected outputs from each batch.

    args:
    split_among_ranks: Optionally split `inputs_dataset` among different global ranks for parallel processing with multiple GPUs, 
        then all gather results from each rank.
    """
    dataloader = DataLoader(
        inputs_dataset,
        batch_size=per_device_batch_size,
        shuffle=False,
        collate_fn=lambda x:x, # dummy collator
        sampler=AsyncContiguousDistributedSampler(inputs_dataset) if split_among_ranks else None,
    )
    outputs_dataset = []
    for inputs_batch in tqdm.tqdm(dataloader, disable=not Accelerator().is_local_main_process):
        outputs_batch = fn(inputs_batch)
        outputs_dataset.extend(outputs_batch)
    if split_among_ranks:
        outputs_dataset = accelerate.utils.gather_object(outputs_dataset)
    return outputs_dataset


class Registry(dict):
    def register(self, module_name: Optional[str] = None) -> Callable:
        def register_fn(module: Callable) -> Callable:
            self[module_name] = module
            return module
        return register_fn
