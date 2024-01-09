NGPUS=4
NCPUS=20
LAUNCH="srun -p llm-safety --quotatype=reserved --gres=gpu:${NGPUS} --cpus-per-task=${NCPUS} accelerate launch --num_processes=${NGPUS} "

output_dir="output/red-teaming/llama-2-7b-fewshot"
victim_model_cls="llama-2-fewshot"
victim_model_name="/mnt/petrelfs/share_data/llm_llama/llama2/llama-2-7b-hf"
back_translator_model_cls="llama-2-fewshot"
back_translator_model_name="/mnt/petrelfs/share_data/llm_llama/llama2/llama-2-7b-hf"

## seeding
# create seed instructions
# python seed.py --output_path "${output_dir}/seed/instruction.jsonl"

# elicit responses from `victim_model`
# PYTHONPATH=. ${LAUNCH} respond.py \
#     --input_path "${output_dir}/seed/instruction.jsonl" \
#     --output_path "${output_dir}/seed/instruction_response.jsonl"  \
#     --victim_model_cls ${victim_model_cls} \
#     --victim_model_name ${victim_model_name}

# score the responses
# PYTHONPATH=. ${LAUNCH} safety_score.py \
#     --input_path "${output_dir}/seed/instruction_response.jsonl" \
#     --output_path "${output_dir}/seed/score.jsonl"

## iteration 1
# back translate
# PYTHONPATH=. ${LAUNCH} back_translate.py \
#     --input_path "${output_dir}/seed/score.jsonl" \
#     --output_path "${output_dir}/iter1/instruction.jsonl" \
#     --back_translator_model_cls ${back_translator_model_cls} \
#     --back_translator_model_name ${back_translator_model_name}


# elicit responses from `victim_model`
PYTHONPATH=. ${LAUNCH} respond.py \
    --input_path "${output_dir}/iter1/instruction.jsonl" \
    --output_path "${output_dir}/iter1/instruction_response.jsonl"  \
    --victim_model_cls ${victim_model_cls} \
    --victim_model_name ${victim_model_name}

# score the responses
PYTHONPATH=. ${LAUNCH} safety_score.py \
    --input_path "${output_dir}/iter1/instruction_response.jsonl" \
    --output_path "${output_dir}/iter1/score.jsonl"
