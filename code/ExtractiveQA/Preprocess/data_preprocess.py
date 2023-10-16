import json
import os
import string
import sys
from io import open

from fol_preprocess import*

print("load train data file")
with open('../../MedEX/Data/new_2hops/Final_2hops_MashQA_kg_train_data.json','r') as f:
    mash_train_data=json.load(f)


train_data=[]

for item,val in enumerate(mash_train_data):
	print("context no: ",item)
	qq=[]
	for qa in mash_train_data[item]["qas"]:
		answers=[]
		answers.append({"text":qa["answers"][0]["text"], "answer_start":qa["answers"][0]["answer_start"]})
		question_text =qa["question"]
		r1,r2,r3,r4,r5,r6 = apply_rules_to_kg(qa['kg_triplets'])
		q = {
			"id":qa["id"],
			"is_impossible": qa["is_impossible"],
			"question": qa["question"],
			"kg_triplets": qa['kg_triplets'],
			"rule_1": r1,
			"rule_2": r2,
			"rule_3": r3,
			"rule_4": r4,
			"rule_5": r5,
			"rule_6": r6,
			"answers": answers
		}
		qq.append(q)
	train ={
			"context":mash_train_data[item]["context"],
			"qas":qq
	}
	train_data.append(train)

file_name='../Data/2_hops/Final_2hops_MashQA_kg_train_data_with_rule.json'
print(file_name)
with open(file_name, 'w') as fp:
	json.dump(train_data, fp, indent=4)


