import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

model_name = "/mnt/petrelfs/share_data/llm_llama/llama2/llama-2-7b-chat-hf"

model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    use_flash_attention_2=True,
)
breakpoint()
tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer.add_bos_token = False # bos_token are in chat template by default
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

prompts = tokenizer.apply_chat_template(
    [{"role": "user", "content": "There's a llama in my garden 😱 What should I do?"}],
    tokenize=False,
)
inputs = tokenizer(prompts, padding=True, return_tensors="pt")

max_new_tokens=256

for i in tqdm.tqdm(range(5)):

    output_tokens = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        attention_mask=inputs["attention_mask"].cuda(),
        do_sample=True,
        max_new_tokens=max_new_tokens,
    )
    output = tokenizer.decode(output_tokens[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    print(output)
    print("\n\n")
