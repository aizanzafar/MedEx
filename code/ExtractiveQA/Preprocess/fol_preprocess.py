import json
import os
import string
import sys
from io import open


"""
# 1. rule of co-coocurance
co_occurs_with(X, Y) ∧ affects(Y, Z) => affects(X, Z)
#  2. Rule of Prevention and Causation:
prevent(X, Y) ∧ causes(Y, Z) => prevent(X, Z)
# 3. Rule of Treatment and Classification:
treat(X, Y) ∧ is_a(Y, Z) => treat(X, Z)
# 4. Rule of Diagnosis and Interaction:
diagnosis(X, Y) ∧ interacts_with(X, Z) => diagnosis(Z, Y)
# 5. Rule of Conjunction:
co_occurs_with(X, Y) ∧ affects(X, Z) => co_occurs_with(Y, Z)
# 6. Rule of Disjunction:
(prevent(X, Y) ∨ causes(Y, Z)) => (prevent(X, Z) ∨ causes(X, Z))

"""

def remove_duplicate(kg_triple):
	res = []
	[res.append(x) for x in kg_triple if x not in res]
	return res


def parse_triple(kg_triplets):
	kg_len = len(kg_triplets)
	empty=['_NAF_H','_NAF_R','_NAF_T']
	if kg_len <=40:
		tt= 40 - kg_len
		for item in range(tt):
			kg_triplets.append(empty)
	return kg_triplets[:40]


def apply_rules_to_kg(kg_triplets):
	co_occurs_triplets = []
	prevent_triplets = []
	treatment_triplets = []
	diagnosis_triplets = []
	conjunction_triplets = []
	disjunction_triplets = []
	# 1.Rule of Co-occurrence: If X co-occurs with Y and Y affects Z, then X affects Z
	for triplet in kg_triplets:
		if triplet[1] == "co-occurs_with":
			for other_triplet in kg_triplets:
				if other_triplet[0] == triplet[2] and other_triplet[1] == "affects":
					if triplet[0] == other_triplet[2]:
						pass
					else:
						co_occurs_triplets.append([triplet[0], "affects", other_triplet[2]])

	# 2.Rule of Prevention and Causation: If X prevents Y and Y causes Z, then X prevents Z
	for triplet in kg_triplets:
		if triplet[1] == "prevents":
			for other_triplet in kg_triplets:
				if other_triplet[0] == triplet[2] and other_triplet[1] == "causes":
					if triplet[0] == other_triplet[2]:
						pass
					else:
						prevent_triplets.append([triplet[0], "prevents", other_triplet[2]])

	# 3.Rule of Treatment and Classification: If X treats Y and Y is a type of Z, then X can be used to treat Z
	for triplet in kg_triplets:
		if triplet[1] == "treats":
			for other_triplet in kg_triplets:
				if other_triplet[0] == triplet[2] and other_triplet[1] == "is_a":
					if triplet[0] == other_triplet[2]:
						pass
					else:
						treatment_triplets.append([triplet[0], "treats", other_triplet[2]])
	# 4.Rule of Diagnosis and Interaction: If X is diagnosed with Y and X interacts with Z, then Z can be used for the diagnosis of Y
	for triplet in kg_triplets:
		if triplet[1] == "diagnosis":
			for other_triplet in kg_triplets:
				if other_triplet[0] == triplet[0] and other_triplet[1] == "interacts_with":
					if other_triplet[2] == triplet[0]:
						pass
					else:
						diagnosis_triplets.append([other_triplet[2], "diagnosis", triplet[0]])
	# 5.Rule of Conjunction .
	for triplet in kg_triplets:
		if triplet[1] == "co-occurs_with":
			for other_triplet in kg_triplets:
				if other_triplet[0] == triplet[0] and other_triplet[1] == "affects":
					if triplet[2] == other_triplet[2]:
						pass
					else:
						conjunction_triplets.append([triplet[2], "co-occurs_with", other_triplet[2]])
	# 5.Rule of disjunction .
	for triple in kg_triplets:
		if triple[1] == "prevents":
			X = triple[0]
			Y = triple[2]
			for other_triple in kg_triplets:
				if other_triple[1] == "causes" and other_triple[0] == Y:
					Z = other_triple[2]
					new_triple1 = [X, "prevents", Z]
					new_triple2 = [X, "causes", Z]
					disjunction_triplets.append(new_triple1)
					disjunction_triplets.append(new_triple2)
	return parse_triple(remove_duplicate(co_occurs_triplets)),parse_triple(remove_duplicate(prevent_triplets)),parse_triple(remove_duplicate(treatment_triplets)),parse_triple(remove_duplicate(diagnosis_triplets)),parse_triple(remove_duplicate(conjunction_triplets)),parse_triple(remove_duplicate(disjunction_triplets))





# ## read mashqa-data
# with open('../Data/MashQA_kg_train_data_10.json','r') as f:
# 	mash_train_data=json.load(f)

# print("data loaded")


# train_data=[]

# for item,val in enumerate(mash_train_data):
# 	print("context no: ",item)
# 	qq=[]
# 	for qa in mash_train_data[item]["qas"]:
# 		answers=[]
# 		answers.append({"text":qa["answers"][0]["text"], "answer_start":qa["answers"][0]["answer_start"]})
# 		question_text =qa["question"]
# 		r1,r2,r3,r4,r5,r6 = apply_rules_to_kg(qa['kg_triplets'])
# 		q = {
# 			"id":qa["id"],
# 			"is_impossible": qa["is_impossible"],
# 			"question": qa["question"],
# 			"kg_triplets": qa['kg_triplets'],
# 			"rule_1": r1,
# 			"rule_2": r2,
# 			"rule_3": r3,
# 			"rule_4": r4,
# 			"rule_5": r5,
# 			"rule_6": r6,
# 			"answers": answers
# 		}
# 		qq.append(q)
# 	train ={
# 			"context":mash_train_data[item]["context"],
# 			"qas":qq
# 	}
# 	train_data.append(train)

# file_name='MashQA_kg_train_data_'+str(len(train_data))+'.json'
# print(file_name)
# with open(file_name, 'w') as fp:
# 	json.dump(train_data, fp, indent=4)
