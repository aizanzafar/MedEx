# from simpletransformers.classification import (ClassificationModel, ClassificationArgs)
# import pandas as pd
# import logging
# import json

# logging.basicConfig(level=logging.INFO)
# transformers_logger = logging.getLogger("transformers")
# transformers_logger.setLevel(logging.WARNING)


# with open('Data/bioasq/new_bioasq_yesno_st.json','r') as f:
#     data=json.load(f)


# train_data=data[:803]
# val_data=data[803:975]
# test_data=data[975:]

# train_df = pd.DataFrame(train_data)
# train_df.columns = ["text_a", "text_b", "labels"]

# eval_df = pd.DataFrame(val_data)
# eval_df.columns = ["text_a", "text_b", "labels"]


# train_args = {
#     'num_train_epochs': 60,
#     'output_dir': "Roberta_bioasq/",
#     'overwrite_output_dir': True,
#     'reprocess_input_data': False,
#     'save_steps':-1,
#     'save_model_every_epoch':False,
#     'train_batch_size': 8
# }

# # Create a ClassificationModel
# # model = ClassificationModel("roberta", "roberta-large", args=train_args)
# model = ClassificationModel("auto", "michiyasunaga/BioLinkBERT-large", args=train_args)


# # Train the model
# model.train_model(train_df)

# # Evaluate the model
# result, model_outputs, wrong_predictions = model.eval_model(eval_df)


# # print("!!!!!!!!!!!!!!!!!! model_outputs !!!!!!!!!!!!")
# # print(model_outputs)

# print("!!!!!!!!!!!!!!!!!! results !!!!!!!!!!!!")
# print(result)

# # print("!!!!!!!!!!!!!!!!!! results !!!!!!!!!!!!")
# # print(wrong_predictions)


from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs
import logging
import json


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


with open('../Data/train_data.json','r') as f:
    train_data=json.load(f)


with open('../Data/val_data.json','r') as f:
    eval_data=json.load(f)


train_args = {
    'learning_rate': 2e-5,
    'num_train_epochs': 30,
    'max_seq_length': 512,
    'doc_stride': 384,
    'output_dir': "bio_dragon_st/",
    'overwrite_output_dir': True,
    'reprocess_input_data': False,
    'save_model_every_epoch': False,
    'save_steps':-1,
    'train_batch_size': 8,
    'eval_batch_size': 4,
    'evaluate_during_training': True,
    'evaluate_during_training_steps': 4000,
    'evaluate_during_training_verbose': True,
    'gradient_accumulation_steps': 4
}


# model = QuestionAnsweringModel("roberta", "roberta-base", args=train_args)

model = QuestionAnsweringModel("bert", "michiyasunaga/dragon", args=train_args)
# Train the model
model.train_model(train_data, eval_data=eval_data)

# Evaluate the model
result, texts = model.eval_model(eval_data)


print("!!!!!!!!!!!!!!!!!!! result !!!!!!!!!!!!!!!!!")
print(result)
