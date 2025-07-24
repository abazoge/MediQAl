import json
import random
import argparse
from datasets import load_dataset, concatenate_datasets, load_from_disk

parser = argparse.ArgumentParser(formatter_class = argparse.RawDescriptionHelpFormatter)
parser.add_argument("--subset", type=str, required=True, help="MediQAl subset (mcqu, mcqm, oeq)")
args = parser.parse_args()
args = vars(args)

def getPrompt(ctx, qst, answ_chs, crt_answ, subset):

	instruction_mcqu = '''You are an experienced medical doctor and independent practitioner. Your task is to answer the following medical multiple-choice question. There is only one correct choice. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCDE. Think step by step before answering.\n\n'''
	instruction_mcqm = '''You are an experienced medical doctor and independent practitioner. Your task is to answer the following medical multiple-choice question. Multiple selections are required; single-choice answers are not accepted. The last line of your response should be of the following format: 'Answer: $LETTERS' (without quotes) where LETTERS are multiple letters of ABCDE, separated by commas (e.g., A,B,C). Think step by step before answering.\n\n'''

	if subset == "mcqu":
		instruction = instruction_mcqu
	elif subset == "mcqm":
		instruction = instruction_mcqm

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


def getPrompt_oeq(ctx, qst, answ):

	instruction = '''You are an experienced medical doctor and independent practitioner. Your task is to answer the following medical question in French by providing a well-structured and concise response. Your response should be of the following format: 'Answer: $TEXT' (without quotes) where TEXT is your final answer written in French. Think step by step before answering.\n\n'''

	def parseElement(context, question, answer):

		if context != None:
			if answer != "":
				return "**Clinical case:** {{context}} \n**Question:** {{question}} \n **Réponse:** {{answer}}" \
				.replace("{{context}}", context) \
				.replace("{{question}}", question) \
				.replace("{{answer}}", answer)
			else:
				return "**Clinical case:** {{context}} \n**Question:** {{question}} \n" \
				.replace("{{context}}", context) \
				.replace("{{question}}", question)
		else:
			if answer != "":
				return "**Question:** {{question}} \n **Réponse:** {{answer}}" \
				.replace("{{question}}", question) \
				.replace("{{answer}}", answer)
			else:
				return "**Question:** {{question}} \n" \
				.replace("{{question}}", question)


	question_answer = parseElement(ctx, qst, answ)

	prompt = instruction + question_answer
	
	return prompt

ds = load_dataset("ANR-MALADES/MediQAl", name=args['subset'])['test']

output_test = []

for d in ds:

	if args['subset'] == "oeq":

		prompt = getPrompt_oeq(
			ctx=d['clinical_case'],
			qst=d['question'],
			answ=d['answer']
		)

		prompt_no_answer = getPrompt_oeq(
			ctx=d['clinical_case'],
			qst=d['question'],
			answ=""
		)

		output_test.append({
			"identifier": d["id"],
			"question_type": d['question_type'],
			"medical_subject": d['medical_subject'],
			"correct_answer": d['answer'],
			
			"prompt": prompt,
			"prompt_no_answer": prompt_no_answer,
		})

	else:

		d_options = ['A', 'B', 'C', 'D', 'E']

		prompt = getPrompt(
			ctx=d['clinical_case'],
			qst=d['question'],
			answ_chs=[{"letter": o, "text": d['answer_'+o.lower()]} for o in d_options],
			crt_answ=d['correct_answers'],
			subset=args['subset']
		)

		prompt_no_answer = getPrompt(
			ctx=d['clinical_case'],
			qst=d['question'],
			answ_chs=[{"letter": o, "text": d['answer_'+o.lower()]} for o in d_options],
			crt_answ="",
			subset=args['subset']
		)

		output_test.append({
			"identifier": d["id"],
			"task_type": d['task'],
			"question_type": d['question_type'],
			"medical_subject": d['medical_subject'],
			"classes": d_options,
			"correct_answer": d['correct_answers'],
			
			"prompt": prompt,
			"prompt_no_answer": prompt_no_answer,
		})

with open(f"test_{args['subset']}.jsonl", "w", encoding="utf-8") as f:
	for item in output_test:
		f.write(json.dumps(item, ensure_ascii=False) + "\n")


