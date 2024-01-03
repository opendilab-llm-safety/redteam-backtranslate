# srun -p llm-safety --quotatype=reserved --gres=gpu:1 --cpus-per-task=8 sh classifier/beavertails_dev.sh
LAUNCH="accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1"

model_name="/mnt/petrelfs/share_data/llm_llama/llama2/llama-2-7b-chat-hf"
output_dir="./output"

run_name="classifier/beavertails_dev"
PYTHONPATH=. $LAUNCH classifier/beavertails_train.py \
    --model_name ${model_name} \
    --sanity_check True \
    --num_proc 1 \
    --peft True \
    --training_args.output_dir "${output_dir}/${run_name}" \
    --training_args.run_name ${run_name}
