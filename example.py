# LLaMA model with KIVI
import warnings

warnings.filterwarnings("ignore")
import torch
import random
from models.llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer
from datasets import load_dataset

# For reproducibility
random.seed(0)
torch.manual_seed(0)
# "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
model_name = "JackFram/llama-160m"

config = LlamaConfig.from_pretrained(model_name)

config.k_bits = 2  # KiVi currently support 2/4 K/V bits
config.v_bits = 2
config.group_size = 32
config.residual_length = 32  # corresponding to the number of recent fp16 tokens
config.use_flash = True

model = LlamaForCausalLM_KIVI.from_pretrained(
    pretrained_model_name_or_path=model_name,
    config=config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
).cuda()

enc = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)

print("Model loaded. Generating prompt...")

prompt = (
    "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n"
    "Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
    "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n\n"
    "Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\n"
    "Answer: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\n"
    "Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n\n"
    "Question: Betty is saving money for a new wallet which costs $100. She has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does she need to buy the wallet?\n"
    "Answer: Betty has 100 / 2 = $50.\n"
    "Her parents gave her $15.\n"
    "Her grandparents gave her 15 * 2 = $30.\n"
    "She has 50 + 15 + 30 = $95.\n"
    "She needs 100 - 95 = $5 more.\n\n"
)

# 加上最后这一个测试问题
prompt += "Question: John takes care of 10 dogs. Each dog takes .5 hours a day to walk and take care of their business. How many hours a week does he spend taking care of dogs?\nAnswer:"


inputs = enc(prompt, return_tensors="pt").input_ids.cuda()

output = model.generate(inputs, max_new_tokens=96)
config_str = f"# prompt tokens: {inputs.shape[1]}, K bit: {config.k_bits}, v_bits: {config.v_bits}, group_size: {config.group_size}, residual_length: {config.residual_length}"

print(prompt + "\n" + "=" * 10 + f"\n{config_str}\n" + "=" * 10 + "\nKiVi Output:")
print(enc.decode(output[0].tolist()[inputs.shape[1] :], skip_special_tokens=True))
