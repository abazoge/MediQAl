import os
import json
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from vllm import LLM, SamplingParams
from datasets import load_from_disk, load_dataset

parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument("--base_model_name", type=str, required=True, help="HuggingFace Hub / Local model name")
parser.add_argument("--input_prompt", type=str, required=True, help="Input for evaluation (0-shot or 3-shot)")
parser.add_argument("--subset", type=str, required=True, help="MediQAl subset (mcqu, mcqm, oeq)")
args = parser.parse_args()
args = vars(args)

THREADS_NBR = 6

base_model_name = args["base_model_name"]
short_base_model_name = base_model_name.split("/")[-1].replace("_","-")

llm = LLM(
    model=base_model_name,
    dtype="bfloat16",
    tensor_parallel_size=2,  # number GPUs
    max_model_len=8192,
)

sampling_params = SamplingParams(
    max_tokens=8192,
    temperature=0.0,
    top_p=1.0,
    top_k=1,
)

if os.path.isdir('./results_zeroshot_batch/') == False:
    os.makedirs('./results_zeroshot_batch/', exist_ok=True)

input_prompt = args["input_prompt"]

subset = args["subset"]

dataset = load_dataset(
        "json",
        data_files={
            "test": f"test_{subset}.jsonl",
        }
    )
print(dataset)

dataset = dataset['test']

prompts = dataset[input_prompt]
identifiers = dataset["identifier"]
correct_answers = dataset["correct_answer"]

all_results = []

batch_size = 2

for i in tqdm(range(0, len(prompts), batch_size)):
    batch_prompts = prompts[i : i + batch_size]
    batch_identifiers = identifiers[i : i + batch_size]
    batch_correct_answers = correct_answers[i : i + batch_size]

    outputs = llm.generate(batch_prompts, sampling_params)

    for prompt, identifier, correct_answer, output in zip(
        batch_prompts, batch_identifiers, batch_correct_answers, outputs
    ):
        gen_text = output.outputs[0].text.strip()
        generated_only = gen_text[len(prompt):].strip() if gen_text.startswith(prompt) else gen_text

        all_results.append({
            "generated_text": generated_only,
            "input_prompt": prompt,
            "identifier": identifier,
            "correct_answer": correct_answer
        })

with open(f"./results_zeroshot_batch/results_{short_base_model_name}_{subset}.json", 'w') as f:
    json.dump(all_results, f, indent=4)
