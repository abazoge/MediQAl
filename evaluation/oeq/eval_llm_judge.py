import re
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from collections import Counter

dataset = ds = load_dataset('ANR-MALADES/MediQAl', name="oeq", trust_remote_code=True)
dataset = dataset['test']

print(dataset)

list_files = [
	["o3", "./api_llm_judge_empty/qcm_llm_judge_o3.jsonl", "./api_llm_judge/results/res_judge_o3.jsonl"],
	["R1", "./api_llm_judge_empty/qcm_llm_judge_R1.jsonl", "./api_llm_judge/results/res_judge_r1.jsonl"],
	["R1-distill-llama-70b", "./api_llm_judge_empty/qcm_llm_judge_r1-distill-llama-70b.jsonl", "./api_llm_judge/results/res_judge_r1-distill-llama-70b.jsonl"],
	["HuatuoGPT-o1-8B", "./api_llm_judge_empty/qcm_llm_judge_FreedomIntelligence-HuatuoGPT-o1-8B.jsonl", "./api_llm_judge/results/res_judge_FreedomIntelligence-HuatuoGPT-o1-8B.jsonl"],
	["FineMedLM-o1", "./api_llm_judge_empty/qcm_llm_judge_hongzhouyu-FineMedLM-o1.jsonl", "./api_llm_judge/results/res_judge_hongzhouyu-FineMedLM-o1.jsonl"],
	["R1-Distill-Llama-8B", "./api_llm_judge_empty/qcm_llm_judge_deepseek-ai-DeepSeek-R1-Distill-Llama-8B.jsonl", "./api_llm_judge/results/res_judge_DeepSeek-R1-Distill-Llama-8B.jsonl"],
	["R1-Distill-Qwen-7B", "./api_llm_judge_empty/qcm_llm_judge_deepseek-ai-DeepSeek-R1-Distill-Qwen-7B.jsonl", "./api_llm_judge/results/res_judge_DeepSeek-R1-Distill-Qwen-7B.jsonl"],
	["GPT-4o", "./api_llm_judge_empty/qcm_llm_judge_gpt4o.jsonl", "./api_llm_judge/results/res_judge_gpt4o.jsonl"],
	["Deepseek-V3", "./api_llm_judge_empty/qcm_llm_judge_deepseekv3.jsonl", "./api_llm_judge/results/res_judge_deepseekv3.jsonl"],
	["Qwen2.5-72b-instruct", "./api_llm_judge_empty/qcm_llm_judge_qwen2.5-72b-instruct.jsonl", "./api_llm_judge/results/res_judge_qwen2.5-72b-instruct.jsonl"],
	["Llama3.3-70b-instruct", "./api_llm_judge_empty/qcm_llm_judge_llama3.3-70b-instruct.jsonl", "./api_llm_judge/results/res_judge_llama3.3-70b-instruct.jsonl"],
	["BioMistral-BioMistral-7B", "./api_llm_judge_empty/qcm_llm_judge_BioMistral-BioMistral-7B.jsonl", "./api_llm_judge/results/res_judge_BioMistral-BioMistral-7B.jsonl"],
	["Llama-3.1-8B-UltraMedical", "./api_llm_judge_empty/qcm_llm_judge_TsinghuaC3I-Llama-3.1-8B-UltraMedical.jsonl", "./api_llm_judge/results/res_judge_TsinghuaC3I-Llama-3.1-8B-UltraMedical.jsonl"],
	["BioMistral-7B-SFT", "./api_llm_judge_empty/qcm_llm_judge_BioMistral-7B-SFT.jsonl", "./api_llm_judge/results/res_judge_BioMistral-7B-SFT.jsonl"],
]


results = []

df = dataset.to_pandas()
df.set_index('id', inplace=True)

for model_name, file_empty, file_res in list_files:

	print(model_name)

	list_u = []
	list_r = []
	list_avg = []

	with open(file_empty, encoding="utf-8") as f1:
		data_empty = [json.loads(line) for line in f1]

	with open(file_res, encoding="utf-8") as f2:
		data_res = [json.loads(line) for line in f2]

	list_identifier = [s['id'] for s in dataset]

	data_empty = [d for d in data_empty if d['id'] in list_identifier]
	data_res = [d for d in data_res if d['key'] in list_identifier]

	for d in data_empty:
		type_question = df.loc[d['id'], 'question_type']
		list_avg.append(0.0)
		if type_question == 'Understanding':
			list_u.append(0.0)
		else:
			list_r.append(0.0)

	for d in data_res:
		type_question = df.loc[d['key'], 'question_type']
		# extract value from prediction
		ANSWER_PATTERN = r"\[\[(\d+)\]\]"
		# print(d['key'])
		try:
			response = d['response']['candidates'][0]['content']['parts'][0]['text']
			match = re.findall(ANSWER_PATTERN, response)

			if len(match) == 1:
				list_avg.append(float(match[-1]))
				if type_question == 'Understanding':
					list_u.append(float(match[-1]))
				else:
					list_r.append(float(match[-1]))
			else:
				list_avg.append(0.0)
				if type_question == 'Understanding':
					list_u.append(0.0)
				else:
					list_r.append(0.0)
		except (KeyError, IndexError, TypeError):
			list_avg.append(0.0)
			if type_question == 'Understanding':
				list_u.append(0.0)
			else:
				list_r.append(0.0)


	results.append({
	"model": model_name,
	"score_understanding": round(np.mean(list_u)*10, 2),
	"score_reasoning": round(np.mean(list_r)*10, 2),
	"score_avg": round(np.mean(list_avg)*10, 2),
	})


with open('result_llm_judge.json', 'w') as json_file:
	json.dump(results, json_file, indent=4)













