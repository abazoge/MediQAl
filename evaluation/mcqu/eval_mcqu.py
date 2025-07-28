import re
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

dataset = ds = load_dataset('ANR-MALADES/MediQAl', name="mcqu", trust_remote_code=True)
dataset = dataset['test']

print(dataset)

list_files = [
	'../mcqu/8B/results_BioMistral-BioMistral-7B-DARE.json',
	'../mcqu/8B/results_deepseek-ai-DeepSeek-R1-Distill-Llama-8B.json',
]

results = []

for f in list_files:

	with open(filepath, encoding="utf-8") as f:
	    data = json.load(f)
	
	results_understanding = []
	results_reasoning = []
	results_average = []
	
	for d in tqdm(data):
	
		df = dataset.to_pandas()
	
		df.set_index('id', inplace=True)
		type_question = df.loc[d['identifier'], 'question_type']
	
		pattern = r"(?i)\*{0,2}(Answer|Réponse)\*{0,2}[ \t]*:[ \t]*\$?\(?([A-E])\)?\$?"
		pattern2 = r"(?i)\*{0,2}(Answer|Réponse)[ \t]*:[ \t]*\*{0,2}[ \t]*\$?\(?([A-E])\)?\$?"
	
		match_all = re.findall(pattern, d['generated_text'])
		match_all2 = re.findall(pattern2, d['generated_text'])
	
		if match_all:
			extracted_answer = match_all[-1][1]
		elif match_all2:
			extracted_answer = match_all2[-1][1]
		else:
			extracted_answer = None
	
		score = 1.0 if extracted_answer == d['correct_letter'] else 0.0
	
		results_average.append(score)
	
		if type_question == "Understanding":
			results_understanding.append(score)
		else:
			results_reasoning.append(score)

	results.append({
		"file_name": f,
		"accuracy_u": round(np.mean(results_understanding), 4)*100,
		"accuracy_r": round(np.mean(results_reasoning), 4)*100,
		"accuracy_avg": round(np.mean(results_average), 4)*100
	})

with open('result_mcqu_8B.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)
