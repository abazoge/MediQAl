import re
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

from rouge import Rouge
import bert_score.score as bertscore
import bleurt.score as bleurtscore
import scipy.stats as stats
import math
import string

import sys
sys.setrecursionlimit(3000)

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os

def clip(value): # clip to a 0-1 value
	return max(0, min(1, value))

def compute_bertscore(predictions, references):
	assert len(predictions) == len(references), "Predictions and references do not have the same size."
	
	bertScore_Precision, bertScore_Recall, bertScore_F1 = bertscore(predictions, references, lang='fr', model_type="roberta-large-mnli", device ='cuda', verbose=True, rescale_with_baseline=True) # roberta-large
	
	bertscores = bertScore_F1.numpy()
	## clip scores to [0,1]
	bertscores = np.array([clip(num) for num in bertscores])

	# return bertscores.mean()
	return bertscores

def compute_bleu(generated_texts, reference_texts):
    if len(generated_texts) != len(reference_texts):
        raise ValueError("The number of generated texts and reference texts must be the same.")
    
    smoothing_function = SmoothingFunction().method1
    bleu_scores = []

    for gen, ref in zip(generated_texts, reference_texts):
        if not gen.strip():  # Handle empty generated text
            bleu_scores.append(0.0)
        else:
            gen_tokens = gen.split()  # Tokenize the generated text
            ref_tokens = ref.split()  # Tokenize the reference text
            score = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothing_function)
            bleu_scores.append(score)

    # average_bleu = sum(bleu_scores) / len(bleu_scores)
    return bleu_scores

def compute_rouge(predictions, references):

	rouge = Rouge() 
	rouge_scores = rouge.get_scores(predictions, references, ignore_empty=True)
					
	rouge1f_scores = []
	
	for i in range(len(references)):
		r1f = rouge_scores[i]["rouge-1"]["f"]
		
		rouge1f_scores.append(r1f)
		
	# for checking comparison with composite
	# return np.array(rouge1f_scores).mean()
	return np.array(rouge1f_scores)

dataset = ds = load_dataset('abazoge/MediQAl', name="qroc", trust_remote_code=True)
dataset = dataset['test']

print(dataset)

list_files = [
	'../oeq/8B/results_BioMistral-BioMistral-7B-DARE_qroc_long.json',
	'../oeq/8B/results_deepseek-ai-DeepSeek-R1-Distill-Llama-8B_qroc_long.json',
	'../oeq/8B/results_deepseek-ai-DeepSeek-R1-Distill-Qwen-7B_qroc_long.json',
	'../oeq/8B/results_FreedomIntelligence-HuatuoGPT-o1-8B_qroc_long.json',
	'../oeq/8B/results_hongzhouyu-FineMedLM-o1_qroc_long.json',
	'../oeq/8B/results_TsinghuaC3I-Llama-3.1-8B-UltraMedical_qroc_long.json'
]

final_list = []

for f in list_files:

	print(f)

	with open(f, encoding="utf-8") as f1:
	    data = json.load(f1)

	list_identifier = [s['id'] for s in dataset]

	data = [d for d in data if d['identifier'] in list_identifier]

	predictions = []
	references = []

	predictions_u = []
	references_u = []

	predictions_r = []
	references_r = []

	list_score_0 = []
	list_score_0_u = []
	list_score_0_r = []

	count_example = 0
	count_example_u = 0
	count_example_r = 0

	for d in tqdm(data):

		ANSWER_PATTERN = r"(?i)(?:Answer|Réponse)\s*:\*{0,2}\s*([^\n]+)"
		match = re.findall(ANSWER_PATTERN, d['generated_text'])

		if match:
			if len(match) == 1:
				extracted_answer = match[-1]
			else:
				extracted_answer = match[-2]
		else:
			extracted_answer = "NA"

		if extracted_answer == "NA" or len(extracted_answer.strip()) <= 0 or re.match(r"^\.+$", extracted_answer):
			list_score_0.append(0.0)

			if d['question_type'] == 'Understanding':
				list_score_0_u.append(0.0)
			else:
				list_score_0_r.append(0.0)
		else:
			predictions.append(extracted_answer)
			references.append(d['answer'])

			if d['question_type'] == 'Understanding':
				predictions_u.append(extracted_answer)
				references_u.append(d['answer'])
				count_example_u+=1
			else:
				predictions_r.append(extracted_answer)
				references_r.append(d['answer'])
				count_example_r+=1

		count_example+=1

	list_u_rouge = compute_rouge(predictions_u, references_u)
	list_u_bleu = compute_bleu(predictions_u, references_u)
	list_u_bertscore = compute_bertscore(predictions_u, references_u)

	rouge_u = np.mean(np.concatenate((list_u_rouge, list_score_0_u)))
	bleu_u = np.mean(np.concatenate((list_u_bleu, list_score_0_u)))
	bertscore_u = np.mean(np.concatenate((list_u_bertscore, list_score_0_u)))

	list_r_rouge = compute_rouge(predictions_r, references_r)
	list_r_bleu = compute_bleu(predictions_r, references_r)
	list_r_bertscore = compute_bertscore(predictions_r, references_r)

	rouge_r = np.mean(np.concatenate((list_r_rouge, list_score_0_r)))
	bleu_r = np.mean(np.concatenate((list_r_bleu, list_score_0_r)))
	bertscore_r = np.mean(np.concatenate((list_r_bertscore, list_score_0_r)))

	list_avg_rouge = compute_rouge(predictions, references)
	list_avg_bleu = compute_bleu(predictions, references)
	list_avg_bertscore = compute_bertscore(predictions, references)

	rouge_avg = np.mean(np.concatenate((list_avg_rouge, list_score_0)))
	bleu_avg = np.mean(np.concatenate((list_avg_bleu, list_score_0)))
	bertscore_avg = np.mean(np.concatenate((list_avg_bertscore, list_score_0)))

	final_list.append({
		"file_name": f,
		"rouge_u": round(rouge_u*100, 2),
		"rouge_r": round(rouge_r*100, 2),
		"rouge_avg": round(rouge_avg*100, 2),
		"bleu_u": round(bleu_u*100, 2),
		"bleu_r": round(bleu_r*100, 2),
		"bleu_avg": round(bleu_avg*100, 2),
		"bertscore_u": round(bertscore_u*100, 2),
		"bertscore_r": round(bertscore_r*100, 2),
		"bertscore_avg": round(bertscore_avg*100, 2),
		})


with open('result_qroc_8B.json', 'w') as json_file:
    json.dump(final_list, json_file, indent=4)