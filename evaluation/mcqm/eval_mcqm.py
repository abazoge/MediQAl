import re
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

def compute_accuracy_exact_match(pred, ref):
    return sorted(list(set(pred))) == sorted(list(set(ref)))

def compute_accuracy_hamming(preds, refs):
    corrects = [True for p in preds if p in refs]
    corrects = sum(corrects)
    total_refs = len(list(set(preds + refs)))
    return corrects / total_refs

dataset = ds = load_dataset('ANR-MALADES/MediQAl', name="mcqm", trust_remote_code=True)
dataset = dataset['test']

print(dataset)

filepath = './mcqm/mcqm_BioMistral-7B-SFT.json'

with open(filepath, encoding="utf-8") as f:
    data = json.load(f)

results_emr_understanding = []
results_hamming_understanding = []
results_emr_reasoning = []
results_hamming_reasoning = []
results_emr_avg = []
results_hamming_avg = []

for d in tqdm(data):

	df = dataset.to_pandas()

	df.set_index('id', inplace=True)
	type_question = df.loc[d['identifier'], 'question_type']

	ANSWER_PATTERN_MULTICHOICE = r"(?i)(?:Answer|Réponse)[ \t]*:[ \t]*\$?([A-E](?:[ \t]*,[ \t]*[A-E])*)\$?"
	ANSWER_PATTERN_MULTICHOICE_bis = r"(?i)(?:Answer|Réponse)[ \t]*:\*\*[ \t]*\$?([A-E](?:[ \t]*,[ \t]*[A-E])*)\$?"

	match = re.findall(ANSWER_PATTERN_MULTICHOICE, d['generated_text'])
	match_bis = re.findall(ANSWER_PATTERN_MULTICHOICE_bis, d['generated_text'])

	if match:
		extracted_answer = match[-1]
	elif match_bis:
		extracted_answer = match_bis[-1]
	else:
		extracted_answer = None


	if extracted_answer != None:
		pred_split = extracted_answer.split(',')
		pred_split = list(set([s.strip() for s in pred_split]))
		ref_split = list(set(d['correct_letter'].split(',')))

	score_emr = compute_accuracy_exact_match(pred_split, ref_split) if extracted_answer != None else 0.0
	score_hamming = compute_accuracy_hamming(pred_split, ref_split) if extracted_answer != None else 0.0

	results_emr_avg.append(score_emr)
	results_hamming_avg.append(score_hamming)

	if type_question == "Understanding":
		results_emr_understanding.append(score_emr)
		results_hamming_understanding.append(score_hamming)
	else:
		results_emr_reasoning.append(score_emr)
		results_hamming_reasoning.append(score_hamming)


emr_understanding = np.mean(results_emr_understanding)
print("EMR")
print("- Understanding", round(emr_understanding, 4)*100)
emr_reasoning = np.mean(results_emr_reasoning)
print("- Reasoning", round(emr_reasoning, 4)*100)
emr_avg = np.mean(results_emr_avg)
print("- Avg", round(emr_avg, 4)*100)


hamming_understanding = np.mean(results_hamming_understanding)
print("Hamming")
print("- Understanding", round(hamming_understanding, 4)*100)
hamming_reasoning = np.mean(results_hamming_reasoning)
print("- Reasoning", round(hamming_reasoning, 4)*100)
hamming_avg = np.mean(results_hamming_avg)
print("- Avg", round(hamming_avg, 4)*100)
