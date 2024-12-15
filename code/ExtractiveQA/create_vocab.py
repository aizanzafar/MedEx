import json
import pandas as pd 
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import random
import pickle
import time 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import transformers
from transformers import RobertaConfig, RobertaTokenizer, BasicTokenizer, RobertaForQuestionAnswering, XLNetTokenizer, XLNetForQuestionAnswering


nltk_stopwords = stopwords.words('english')


### load berttokenizer and save vocab 
# tokenizer = XLNetTokenizer.from_pretrained("xlnet-large-cased")

# tokenizer.save_vocabulary('./xlnet/')
# print("tokenizer vocab")
# tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

# tokenizer.save_vocabulary('vocab/roberta')


entity_list = []
# relation_list = []

print("load train data file")
with open('Data/1_hops/Final_MashQA_kg_train_data.json','r') as f:
    train_data=json.load(f)

for item,val in enumerate(train_data):
    for qa in train_data[item]["qas"]:
        for t in qa['kg_triplets']:
            for c in t:
                entity_list.append(c)

print("load val data file")
with open('Data/1_hops/Final_MashQA_kg_val_data.json','r') as f:
    val_data=json.load(f)

for item,val in enumerate(val_data):
    for qa in val_data[item]["qas"]:
        for t in qa['kg_triplets']:
            for c in t:
                entity_list.append(c)

print(len(entity_list))

# entity_list = pharmkg['Entity1_name'].tolist()
# entity_list += pharmkg['Entity2_name'].tolist()
entity_list += ['_NAF_H', '_NAF_T', '_NAF_R']


entity_list = list(set(entity_list))
print(len(entity_list))

entity_dic = {}

for i,item in enumerate(entity_list):
    entity_dic[item] = i

pickle.dump(entity_dic,open('vocab/roberta/entity_vocab_dic.pkl','wb'))
print("done")
# relation_list = pharmkg['relationship_type'].tolist()
# relation_list += ['_NAF_R']

# relation_list = set(relation_list)
# entity_list = set(entity_list)

# print("ee len: ", len(entity_list))
# print('rr len: ',len(relation_list))

# relation_dic = {}
# entity_dic = {}

# for i,item in enumerate(relation_list):
#     relation_dic[item] = i

# for i,item in enumerate(entity_list):
#     entity_dic[item] = i

# pickle.dump(relation_dic,open('xlnet/pharmKG_relation_dic.pkl','wb'))
# pickle.dump(entity_dic,open('xlnet/pharmKG_entity_dic.pkl','wb'))


# print("xlnet vocab load\n")
# f_v = open('xlnet/vocab.txt')
# vocab = f_v.read().split('\n')
# print(vocab[-10:])



# # entity_l = [item.split('_') if '_' in str(item) else [item] for item in entity_list]
# # entity_l = [i for item in entity_l for i in item if i not in nltk_stopwords]

# # entity_list = set(list(entity_list) + list(set(entity_l)))
# # entity_list = set(list(entity_list))

# print("entity list len :",len(entity_list))

# new_entity_list = []

# for item in entity_list:
#     # print(item)
#     item = '_'.join(str(item).split(' '))
#     # print(item)
#     if item in vocab:
#         pass
#     else:
#         new_entity_list.append(item)

# new_vocab = vocab[:-1] + list(new_entity_list)
# print("new vocab\n")
# print(new_vocab[-10:])
# print("new vocab len: ",len(new_vocab))

# new_vocab = list(set(new_vocab))
# print("new vocab len: ",len(new_vocab))

# f_voc = open('roberta/pharmKG_vocab.txt','w')

# for i,item in enumerate(new_vocab):
#     # print("new_vocab:", item)
#     f_voc.write(str(item))
#     f_voc.write('\n')
# f_voc.close()
