import json
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
import re

def getPrompt_judge(question, answer_ref, answer_a):
	return '''
	[System]
	Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the French medical question displayed below. You evaluation should consider clinical correctness, factual coverage and the impact of differences between the answers on patient safety and care. You will be given a reference answer (Expert-provided answer) and the assistant's answer (Model-generated answer). Your job is to evaluate how closely a Model-generated answer aligns with an Expert-provided answer. Base your judgement only on the Expert's provided answer, and never rely on your own medical knowledge or external resources. Begin your evaluation by comparing the assistant's answer with the reference answer. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format "[[rating]], for example: "Rating: [[5]]".

	[Medical Question]
	{{question}}

	[The Start of Expert Answer]
	{{answer_ref}}
	[The End of Expert Answer]

	[The Start of Assistant Answer]
	{{answer_a}}
	[The End of Assistant Answer]''' \
	.replace("{{question}}", question) \
	.replace("{{answer_ref}}", answer_ref) \
	.replace("{{answer_a}}", answer_a) \

def load_json(f, dataset):

	with open(f, encoding="utf-8") as f1:
		data = json.load(f1)

	list_identifier = [s['id'] for s in dataset]

	data = [d for d in data if d['identifier'] in list_identifier]

	output = []

	for d in tqdm(data):

		df = dataset.to_pandas()

		df.set_index('id', inplace=True)

		identifier = d['identifier']
		type_question = df.loc[identifier, 'question_type']
		question = df.loc[identifier, 'question']
		answer_ref = df.loc[identifier, 'answer']

		ANSWER_PATTERN = r"(?i)(?:Answer|Réponse)\s*:\*{0,2}\s*([^\n]+)(?=(?:\n| )*(?:Answer|Réponse)\s*:|\Z)"

		response = d['generated_text']

		segments = re.split(r"(?i)(?=Answer\s*:\*{0,2}\s*|Réponse\s*:\*{0,2}\s*)", response)

		# Extract only those that start with "Answer:"
		answers = []
		for seg in segments:
			if re.match(r"(?i)(?:Answer|Réponse)\s*:\*{0,2}", seg):
				# Remove the label and strip whitespace
				cleaned = re.sub(r"(?i)(?:Answer|Réponse)\s*:\*{0,2}", "", seg).strip()
				answers.append(cleaned)

		if len(answers)==0:
			extracted_answer = ""
		else:
			extracted_answer = answers[-1]

		output.append({"id": identifier, "question": question, "answer_a": extracted_answer, "answer_ref": answer_ref})

	return output


dataset = load_dataset('ANR-MALADES/MediQAl', name="oeq")
dataset = dataset['test']

list_files = [
	"oeq_BioMistral-7B-SFT.json",
	# "oeq_BioMistral-7B-SFT.json",
]

for file in list_files:

	res = load_json(file, dataset)

	res_prompts = []
	res_empty = []

	for r in res:
		if r['answer_a'] == "":
			res_empty.append(r)
		else:
			res_prompts.append({'custom_id': r["id"], 'text': getPrompt_judge(r["question"], r["answer_ref"], r["answer_a"])})

	final_prompts = []

	for d in res_prompts:

		final_prompts.append({
			"key": d['custom_id'],
			"request": {
				"contents": [{"parts": [{"text": d['text']}]}]
			}
		})

	model_name = file.split('_')[1].replace('.json', '')
	with open(f'./api_llm_judge/oeq_llm_judge_{model_name}.jsonl', 'w') as f:
		for item in final_prompts:
			f.write(json.dumps(item) + '\n')

	model_name = file.split('_')[1].replace('.json', '')
	with open(f'./api_llm_judge_empty/oeq_llm_judge_{model_name}.jsonl', 'w') as f:
		for item in res_empty:
			f.write(json.dumps(item) + '\n')

