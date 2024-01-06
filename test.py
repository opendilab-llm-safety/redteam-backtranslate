# PYTHONPATH=. srun -p llm-safety --quotatype=reserved --gres=gpu:8 --cpus-per-task=64 accelerate launch --num_processes=8 test.py 
# PYTHONPATH=. srun -p llm-safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=8 python3 test.py

from datasets import load_dataset

from src.victims import VictimLlama2Chat
from src.utils import batchify_apply, print_local_main


generation_configs = {"do_sample":False, "max_new_tokens":512}
target_lm = VictimLlama2Chat(
    model_name="/mnt/petrelfs/share_data/llm_llama/llama2/llama-2-7b-chat-hf", 
    generation_configs=generation_configs
)

dataset = load_dataset('PKU-Alignment/BeaverTails', split='30k_test')

generations = batchify_apply(
    target_lm.respond, 
    dataset['prompt'][:500],
    per_device_batch_size=8, 
    split_among_ranks=True,
)
print_local_main(len(generations))
breakpoint()
