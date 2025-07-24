from datasets import load_dataset, concatenate_datasets, load_from_disk
import json
import random

def getPrompt_oeq(ctx, qst, answ_chs, crt_answ):

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

def getPrompt_mcqu(ctx, qst, answ_chs, crt_answ):

	instruction = '''You are an experienced medical doctor and independent practitioner. Your task is to answer the following medical multiple-choice question. There is only one correct choice. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCDE. Think step by step before answering.\n\n'''

	def parseElement(context, question, answer_choices, correct_answer_letter):

		answer_choices = " \n ".join([f"({a['letter'].upper()}) {a['text']}" for a in answer_choices])

		if context != None:
			if correct_answer_letter != "":
				return "**Clinical case:** {{context}} \n**Question:** {{question}} \n {{answer_choices}} \n **Answer:** {{correct_answer_letter}}" \
				.replace("{{context}}", context) \
				.replace("{{question}}", question) \
				.replace("{{answer_choices}}", answer_choices) \
				.replace("{{correct_answer_letter}}", correct_answer_letter)
			else:
				return "**Clinical case:** {{context}} \n**Question:** {{question}} \n {{answer_choices}}\n" \
				.replace("{{context}}", context) \
				.replace("{{question}}", question) \
				.replace("{{answer_choices}}", answer_choices)
		else:
			if correct_answer_letter != "":
				return "**Question:** {{question}} \n {{answer_choices}} \n **Answer:** {{correct_answer_letter}}" \
				.replace("{{question}}", question) \
				.replace("{{answer_choices}}", answer_choices) \
				.replace("{{correct_answer_letter}}", correct_answer_letter)
			else:
				return "**Question:** {{question}} \n {{answer_choices}} \n" \
				.replace("{{question}}", question) \
				.replace("{{answer_choices}}", answer_choices)


	question_answer = parseElement(ctx, qst, answ_chs, crt_answ)

	prompt = instruction + question_answer
	
	return prompt


def getPrompt_mcqm(ctx, qst, answ_chs, crt_answ):

	instruction = '''You are an experienced medical doctor and independent practitioner. Your task is to answer the following medical multiple-choice question. Multiple selections are required; single-choice answers are not accepted. The last line of your response should be of the following format: 'Answer: $LETTERS' (without quotes) where LETTERS are multiple letters of ABCDE, separated by commas (e.g., A,B,C). Think step by step before answering.\n\n'''

	def parseElement(context, question, answer_choices, correct_answer_letter):

		answer_choices = " \n ".join([f"({a['letter'].upper()}) {a['text']}" for a in answer_choices])

		if context != None:
			if correct_answer_letter != "":
				return "**Clinical case:** {{context}} \n**Question:** {{question}} \n {{answer_choices}} \n **Answer:** {{correct_answer_letter}}" \
				.replace("{{context}}", context) \
				.replace("{{question}}", question) \
				.replace("{{answer_choices}}", answer_choices) \
				.replace("{{correct_answer_letter}}", correct_answer_letter)
			else:
				return "**Clinical case:** {{context}} \n**Question:** {{question}} \n {{answer_choices}}\n" \
				.replace("{{context}}", context) \
				.replace("{{question}}", question) \
				.replace("{{answer_choices}}", answer_choices)
		else:
			if correct_answer_letter != "":
				return "**Question:** {{question}} \n {{answer_choices}} \n **Answer:** {{correct_answer_letter}}" \
				.replace("{{question}}", question) \
				.replace("{{answer_choices}}", answer_choices) \
				.replace("{{correct_answer_letter}}", correct_answer_letter)
			else:
				return "**Question:** {{question}} \n {{answer_choices}} \n" \
				.replace("{{question}}", question) \
				.replace("{{answer_choices}}", answer_choices)


	question_answer = parseElement(ctx, qst, answ_chs, crt_answ)

	prompt = instruction + question_answer
	
	return prompt

ds_mcqu = load_dataset("ANR-MALADES/MediQAl", name="mcqu")
ds_mqcm = load_dataset("ANR-MALADES/MediQAl", name="mcqm")

print(ds_mcqu)
print(ds_mqcm)

train = []
validation = []

#### MCQU subset

for d in ds_mcqu['train']:

	# MCQU prompt
	d_options = ['A', 'B', 'C', 'D', 'E']
	prompt = getPrompt_mcqu(
		ctx=d['clinical_case'],
		qst=d['question'],
		answ_chs=[{"letter": o, "text": d['answer_'+o.lower()]} for o in d_options],
		crt_answ=d['correct_answers']
	)
	train.append({"text": prompt})

	# OEQ prompt converted from MCQU
	prompt = getPrompt_oeq(
		ctx=d['clinical_case'],
		qst=d['question'],
		answ_chs=[{"letter": o, "text": d['answer_'+o.lower()]} for o in d_options],
		crt_answ=d['correct_answers']
	)
	train.append({"text": prompt})

for d in ds_mcqu['validation']:
	# MCQU prompt
	d_options = ['A', 'B', 'C', 'D', 'E']
	prompt = getPrompt_mcqu(
		ctx=d['clinical_case'],
		qst=d['question'],
		answ_chs=[{"letter": o, "text": d['answer_'+o.lower()]} for o in d_options],
		crt_answ=d['correct_answers']
	)
	validation.append({"text": prompt})

	# OEQ prompt converted from MCQU
	prompt = getPrompt_oeq(
		ctx=d['clinical_case'],
		qst=d['question'],
		answ_chs=[{"letter": o, "text": d['answer_'+o.lower()]} for o in d_options],
		crt_answ=d['correct_answers']
	)
	validation.append({"text": prompt})

#### MCQM subset

for d in ds_mqcm['train']:

	# MCQM prompt
	d_options = ['A', 'B', 'C', 'D', 'E']
	prompt = getPrompt_mcqm(
		ctx=d['clinical_case'],
		qst=d['question'],
		answ_chs=[{"letter": o, "text": d['answer_'+o.lower()]} for o in d_options],
		crt_answ=d['correct_answers']
	)
	train.append({"text": prompt})

	# OEQ prompt converted from MCQM
	prompt = getPrompt_oeq(
		ctx=d['clinical_case'],
		qst=d['question'],
		answ_chs=[{"letter": o, "text": d['answer_'+o.lower()]} for o in d_options],
		crt_answ=d['correct_answers']
	)
	train.append({"text": prompt})

for d in ds_mqcm['validation']:
	# MCQM prompt
	d_options = ['A', 'B', 'C', 'D', 'E']
	prompt = getPrompt_mcqm(
		ctx=d['clinical_case'],
		qst=d['question'],
		answ_chs=[{"letter": o, "text": d['answer_'+o.lower()]} for o in d_options],
		crt_answ=d['correct_answers']
	)
	validation.append({"text": prompt})

	# OEQ prompt converted from MCQM
	prompt = getPrompt_oeq(
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
