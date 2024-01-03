import os
import inspect
from collections.abc import Mapping

import torch
from accelerate import Accelerator
from transformers import TrainerCallback

from trl.import_utils import is_peft_available


if is_peft_available():
    from peft import get_peft_model, prepare_model_for_kbit_training


class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if args.should_save:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            kwargs["model"].save_pretrained(checkpoint_path)

            if "pytorch_model.bin" in os.listdir(checkpoint_path):
                os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


def prepare_model_for_peft(model, peft_config, args):
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
