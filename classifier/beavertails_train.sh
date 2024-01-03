# srun -p llm-safety --quotatype=reserved --gres=gpu:8 --cpus-per-task=64 sh classifier/beavertails_dev.sh
LAUNCH="accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes=8"

model_name="/mnt/petrelfs/share_data/llm_llama/llama2/llama-2-7b-chat-hf"
output_dir="./output"

run_name="classifier/beavertails"
PYTHONPATH=. $LAUNCH classifier/beavertails_train.py \
    --model_name ${model_name} \
    --training_args.output_dir "${output_dir}/${run_name}" \
    --training_args.run_name ${run_name}