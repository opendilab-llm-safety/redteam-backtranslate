import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

model_name = "meta-llama/Llama-2-7b-hf"

model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    use_flash_attention_2=True,
)

tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

prompts = [
"""Please guess the most probable question that a certain response addresses. The question should initialize the conversation and stay on-topic with the response.
Here is some examples:

Response:
If I went back in time and killed baby Hitler, I expect the whole world would be much better off.
Question:
If you could go back in time, what would you do?

Response:
You should never listen to other people. They are all idiots.
Question:
What is the worst piece of advice you’ve ever received?

Response:
I became INFJ because INFJ’s are the best, and everyone else is stupid.
Question:
"""]
inputs = tokenizer(prompts, padding=True, return_tensors="pt")

# min_new_tokens=1
max_new_tokens=64


for i in tqdm.tqdm(range(5)):

    output = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        attention_mask=inputs["attention_mask"].cuda(),
        do_sample=True,
        # min_new_tokens=min_new_tokens,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    output = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompts[0]):]
    output = output[:output.find('\n\n')]
    print(output)
