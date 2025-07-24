import os
import json
import torch
import argparse
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument("--base_model_name", type=str, required=True, help="HuggingFace Hub / Local model name")
parser.add_argument("--input_prompt", type=str, required=True, help="Input for evaluation (0-shot or 3-shot)")
args = parser.parse_args()
args = vars(args)

THREADS_NBR = 6

base_model_name = args["base_model_name"]
short_base_model_name = base_model_name.split("/")[-1].replace("_","-")

model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="cuda", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 

if os.path.isdir('./results_zeroshot_batch/') == False:
    os.makedirs('./results_zeroshot_batch/', exist_ok=True)

input_prompt = args["input_prompt"]

# dataset = load_from_disk(f"../data/local_hf_mcqu")
dataset = load_dataset("ANR-MALADES/MediQAl", name="mcqu")
print(dataset)

dataset = dataset['test']

torch.set_default_device("cuda")

all_results = []

batch_size = 24  # Adjust based on GPU memory

def collate_fn(batch):
    inputs = tokenizer([d[input_prompt] for d in batch], return_tensors="pt", padding=True, truncation=True)
    identifiers = [d["identifier"] for d in batch]
    correct_letters = [d[f"prompt"][-1] for d in batch]
    input_prompts = [d[input_prompt] for d in batch]
    return inputs, identifiers, correct_letters, input_prompts

dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

with torch.no_grad():

    for inputs, identifiers, correct_letters, input_prompts in tqdm(dataloader):
        input_ids = inputs["input_ids"].to(torch.cuda.current_device())
        attention_mask = inputs["attention_mask"].to(torch.cuda.current_device())

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=2048,
            num_beams=1,
            do_sample=False
        )

        outputs = outputs.to("cpu")
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for gen_text, prompt, identifier, correct_letter in zip(generated_texts, input_prompts, identifiers, correct_letters):
            all_results.append({
                "generated_text": gen_text[len(prompt):].strip(),
                "input_prompt": prompt,
                "identifier": identifier,
                "correct_letter": correct_letter
            })

with open(f"./results_zeroshot_batch/results_{short_base_model_name}_mcqu.json", 'w') as f:
    json.dump(all_results, f, indent=4)
