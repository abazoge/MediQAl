from datasets import load_dataset, concatenate_datasets, load_from_disk
import json
import random

def getPrompt(ctx, qst, answ_chs, crt_answ):

	instruction = '''You are an experienced medical doctor and independent practitioner. Your task is to answer the following medical question in French by providing a well-structured and concise response. The last line of your response should be of the following format: 'Answer: $TEXT' (without quotes) where TEXT is your final answer in French. Think step by step before answering.\n\n'''

	def parseElement(context, question, answer_choices, correct_answer_letter):

		answer_text = [a['text'] for r in correct_answer_letter.split(',') for a in answer_choices if a.get('letter') == r]
		answer_string = ". ".join(answer_text)

		if context != None:
			if correct_answer_letter != "":
				return "**Clinical case:** {{context}} \n**Question:** {{question}} \n **Answer:** {{correct_answer}}" \
				.replace("{{context}}", context) \
				.replace("{{question}}", question) \
				.replace("{{correct_answer}}", answer_string)
		else:
			if correct_answer_letter != "":
				return "**Question:** {{question}} \n **Answer:** {{correct_answer}}" \
				.replace("{{question}}", question) \
				.replace("{{correct_answer}}", answer_string)


	question_answer = parseElement(ctx, qst, answ_chs, crt_answ)

	prompt = instruction + question_answer
	
	return prompt

ds_qcu = load_dataset("ANR-MALADES/MediQAl", name="mcqu")
ds_qcm = load_dataset("ANR-MALADES/MediQAl", name="mcqm")

print(ds_qcu)
print(ds_qcm)

print(ds_qcm["train"][0]['prompt'])

mediqal_qcu = load_from_disk("../../mediqal/local_hf_mediqal_qcu")
mediqal_qcm = load_from_disk("../../mediqal/local_hf_mediqal_qcm")

train = []
validation = []

for d in ds_qcu['train']:
	train.append({'text': d['prompt']})

for d in ds_qcu['validation']:
	validation.append({'text': d['prompt']})

for d in ds_qcm['train']:
	train.append({'text': d['prompt']})

for d in ds_qcm['validation']:
	validation.append({'text': d['prompt']})

for d in mediqal_qcu['train']:
	d_options = ['A', 'B', 'C', 'D', 'E']
	prompt = getPrompt(
		ctx=d['clinical_case'],
		qst=d['question'],
		answ_chs=[{"letter": o, "text": d['answer_'+o.lower()]} for o in d_options],
		crt_answ=d['correct_answers']
	)
	train.append({"text": prompt})

for d in mediqal_qcu['validation']:
	d_options = ['A', 'B', 'C', 'D', 'E']
	prompt = getPrompt(
		ctx=d['clinical_case'],
		qst=d['question'],
		answ_chs=[{"letter": o, "text": d['answer_'+o.lower()]} for o in d_options],
		crt_answ=d['correct_answers']
	)
	validation.append({"text": prompt})

for d in mediqal_qcm['train']:
	d_options = ['A', 'B', 'C', 'D', 'E']
	prompt = getPrompt(
		ctx=d['clinical_case'],
		qst=d['question'],
		answ_chs=[{"letter": o, "text": d['answer_'+o.lower()]} for o in d_options],
		crt_answ=d['correct_answers']
	)
	train.append({"text": prompt})

for d in mediqal_qcm['validation']:
	d_options = ['A', 'B', 'C', 'D', 'E']
	prompt = getPrompt(
		ctx=d['clinical_case'],
		qst=d['question'],
		answ_chs=[{"letter": o, "text": d['answer_'+o.lower()]} for o in d_options],
		crt_answ=d['correct_answers']
	)
	validation.append({"text": prompt})

print(len(train))
print(len(validation))

random.seed(42)  # Set the seed
random.shuffle(train)
random.shuffle(validation)


with open("train.jsonl", "w", encoding="utf-8") as f:
    for item in train:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open("validation.jsonl", "w", encoding="utf-8") as f:
    for item in validation:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")