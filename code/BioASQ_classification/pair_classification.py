import json
import torch
import pickle
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, AdamW, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, RobertaForSequenceClassification, RobertaTokenizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from fol_model import*

##data preprocessing
with open('../Data/bioasq/2hops_bioasq_yesno_kg_data_with_rule.json','r') as f: ##bioasq_10b_yesno.json
    data= json.load(f)


vocab_path="../vocab/roberta/bioasq_entity_vocab_dic.pkl"
vocab_dic = pickle.load(open(vocab_path,'rb'))


train_data=data[:950]
val_data=data[950:975]
test_data=data[975:]
# #### biobert best case
# train_data=data[:750]
# val_data=data[750:950]
# test_data=data[950:]

# train_data=data[:861]
# val_data=data[861:1033]
# test_data=data[1033:]

# Define a function for preprocessing the KG triple
def parse_triple(kg_triplets,max_triple):
    kg_len = len(kg_triplets)
    empty=['_NAF_H','_NAF_R','_NAF_T']
    if kg_len <= max_triple:
        tt= max_triple - kg_len
        for item in range(tt):
            kg_triplets.append(empty)
    return kg_triplets

def transform_triple_to_hrt(triple,vocab_dic):
    """ Transforms triple-idx (as a whole) to h/r/t format """
    h, r, t = triple
    return [vocab_dic[h], vocab_dic[r], vocab_dic[t]]

def convert_tokens_to_id(kg,vocab_dic,max_triple):
    kg = parse_triple(kg,max_triple)
    kg = [transform_triple_to_hrt(triple,vocab_dic)  for triple in kg]
    return kg


# Define a function for preprocessing the data
def preprocess_data(data):
    contexts = []
    questions = []
    labels = []
    kg_triples=[]
    rule_1=[]
    rule_2=[]
    rule_3=[]
    rule_4=[]
    rule_5=[]
    rule_6=[]

    for item in data:
        context = item['context']
        question = item['question']
        label = item['exact answer']  # 'yes' or 'no'
        kg_input = item["kg_triple"]
        fol_rule_1 = item['rule_1']
        fol_rule_2 = item['rule_2']
        fol_rule_3 = item['rule_3']
        fol_rule_4 = item['rule_4']
        fol_rule_5 = item['rule_5']
        fol_rule_6 = item['rule_6']

        # Tokenize and encode the text data with padding
        tokenized_data = tokenizer(
                    context,
                    question,
                    padding=True,
                    truncation="only_first",  # Use "only_first" strategy
                    return_tensors="pt",
                    max_length=512
                )

        input_ids = tokenized_data["input_ids"].squeeze()
        attention_mask = tokenized_data["attention_mask"].squeeze()

        contexts.append(input_ids)
        questions.append(attention_mask)

        # Convert labels to binary format: 'yes' -> 1, 'no' -> 0
        label = 1 if label == 'yes' else 0
        labels.append(label)

        ##### kg input preprocessing ######
        max_triple = 200
        kg_enc = kg_input[:max_triple]
        kg_enc_input = convert_tokens_to_id(kg_enc,vocab_dic,max_triple)
        rule_1_enc = convert_tokens_to_id(fol_rule_1,vocab_dic,40)
        rule_2_enc = convert_tokens_to_id(fol_rule_2,vocab_dic,40)
        rule_3_enc = convert_tokens_to_id(fol_rule_3,vocab_dic,40)
        rule_4_enc = convert_tokens_to_id(fol_rule_4,vocab_dic,40)
        rule_5_enc = convert_tokens_to_id(fol_rule_5,vocab_dic,40)
        rule_6_enc = convert_tokens_to_id(fol_rule_6,vocab_dic,40)

        kg_triples.append(kg_enc_input)
        rule_1.append(rule_1_enc)
        rule_2.append(rule_2_enc)
        rule_3.append(rule_3_enc)
        rule_4.append(rule_4_enc)
        rule_5.append(rule_5_enc)
        rule_6.append(rule_6_enc)

    # Determine the maximum sequence length among all examples
    max_seq_length = max(len(seq) for seq in contexts)

    # Pad sequences to the maximum length
    contexts = [torch.cat([seq, torch.zeros(max_seq_length - len(seq)).long()]) for seq in contexts]
    questions = [torch.cat([seq, torch.zeros(max_seq_length - len(seq)).long()]) for seq in questions]
    # all_kg_inputs= [torch.tensor([f for f in kg_triples], dtype=torch.long)]
    # rule_1_inputs = torch.tensor([f for f in rule_1], dtype=torch.long)
    # rule_2_inputs = torch.tensor([f for f in rule_2], dtype=torch.long)
    # rule_3_inputs = torch.tensor([f for f in rule_3], dtype=torch.long)
    # rule_4_inputs = torch.tensor([f for f in rule_4], dtype=torch.long)
    # rule_5_inputs = torch.tensor([f for f in rule_5], dtype=torch.long)
    # rule_6_inputs = torch.tensor([f for f in rule_6], dtype=torch.long)

    return torch.stack(contexts), torch.stack(questions), torch.tensor(labels), torch.tensor(kg_triples), torch.tensor(rule_1), torch.tensor(rule_2), torch.tensor(rule_3), torch.tensor(rule_4), torch.tensor(rule_5), torch.tensor(rule_6)


##### Load the tokenizer
# tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
tokenizer = AutoTokenizer.from_pretrained('michiyasunaga/BioLinkBERT-large')
# tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

# Preprocess the training data
train_contexts, train_questions, train_labels, train_kg, train_rule_1, train_rule_2, train_rule_3, train_rule_4, train_rule_5, train_rule_6 = preprocess_data(train_data)

# Create PyTorch DataLoader for training data
train_dataset = TensorDataset(train_contexts, train_questions, train_labels, train_kg, train_rule_1, train_rule_2, train_rule_3, train_rule_4, train_rule_5, train_rule_6)
batch_size = 2  # Adjust as needed
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Similarly, preprocess the validation and test data
val_contexts, val_questions, val_labels, val_kg, val_rule_1, val_rule_2, val_rule_3, val_rule_4, val_rule_5, val_rule_6 = preprocess_data(val_data)
test_contexts, test_questions, test_labels, test_kg, test_rule_1, test_rule_2, test_rule_3, test_rule_4, test_rule_5, test_rule_6 = preprocess_data(test_data)

val_dataset = TensorDataset(val_contexts, val_questions, val_labels, val_kg, val_rule_1, val_rule_2, val_rule_3, val_rule_4, val_rule_5, val_rule_6)
test_dataset = TensorDataset(test_contexts, test_questions, test_labels, test_kg, test_rule_1, test_rule_2, test_rule_3, test_rule_4, test_rule_5, test_rule_6)

val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle validation
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle test

# Initialize the BioBERT model for sequence classification
# model = BertForSequenceClassification.from_pretrained("dmis-lab/biobert-v1.1", num_labels=2)
# model = AutoModelForSequenceClassification.from_pretrained('michiyasunaga/BioLinkBERT-large', num_labels=2)
# model = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=2)

# model_name="dmis-lab/biobert-v1.1"
model_name='michiyasunaga/BioLinkBERT-large'
config = AutoConfig.from_pretrained(model_name)
model = QA_model(model_name, config, len(vocab), t_embed=300)

# Define the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Move the model to the appropriate device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training parameters
num_epochs = 10

# Initialize variables to keep track of the best validation accuracy and the corresponding model
best_val_accuracy = 0.0
best_model = None

# Training loop
for epoch in range(num_epochs):
    # Training
    model.train()
    total_loss = 0.0
    predictions, true_labels = [], []

    for batch in train_dataloader:
        input_ids, attention_mask, batch_labels, kg_input, rule_1, rule_2, rule_3, rule_4, rule_5, rule_6 = batch
        input_ids, attention_mask, batch_labels, kg_input, rule_1, rule_2, rule_3, rule_4, rule_5, rule_6 = input_ids.to(device), attention_mask.to(device), batch_labels.to(device), kg_input.to(device), rule_1.to(device), rule_2.to(device), rule_3.to(device), rule_4.to(device), rule_5.to(device), rule_6.to(device)

        optimizer.zero_grad()
        outputs = model(kg_input=kg_input, input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels, rule_1=rule_1, rule_2=rule_2, rule_3=rule_3, rule_4=rule_4, rule_5=rule_5, rule_6=rule_6)
        # print("!!!!!!!!! model outputs\n", outputs)
        loss = outputs.loss
        total_loss += loss.item()

        logits = outputs.logits
        predictions.extend(logits.argmax(dim=1).tolist())
        true_labels.extend(batch_labels.tolist())

        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(train_dataloader)
    accuracy = accuracy_score(true_labels, predictions)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}, Training Accuracy: {accuracy:.4f}")

    # Validation
    model.eval()
    val_predictions, val_true_labels = [], []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_mask, batch_labels, kg_input, rule_1, rule_2, rule_3, rule_4, rule_5, rule_6 = batch
            input_ids, attention_mask, batch_labels, kg_input, rule_1, rule_2, rule_3, rule_4, rule_5, rule_6 = input_ids.to(device), attention_mask.to(device), batch_labels.to(device), kg_input.to(device), rule_1.to(device), rule_2.to(device), rule_3.to(device), rule_4.to(device), rule_5.to(device), rule_6.to(device)

            outputs = model(kg_input=kg_input, input_ids=input_ids, attention_mask=attention_mask, rule_1=rule_1, rule_2=rule_2, rule_3=rule_3, rule_4=rule_4, rule_5=rule_5, rule_6=rule_6)
            logits = outputs.logits
            val_predictions.extend(logits.argmax(dim=1).tolist())
            val_true_labels.extend(batch_labels.tolist())

    val_accuracy = accuracy_score(val_true_labels, val_predictions)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Save the best model based on validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model = model

# # Save the best model
# if best_model:
#     best_model.save_pretrained("best_bio_classification_model")

# Evaluation on the test set using the best model
if best_model:
    best_model.eval()
    test_predictions, test_true_labels = [], []

    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, batch_labels, kg_input, rule_1, rule_2, rule_3, rule_4, rule_5, rule_6 = batch
            input_ids, attention_mask, batch_labels, kg_input, rule_1, rule_2, rule_3, rule_4, rule_5, rule_6 = input_ids.to(device), attention_mask.to(device), batch_labels.to(device), kg_input.to(device), rule_1.to(device), rule_2.to(device), rule_3.to(device), rule_4.to(device), rule_5.to(device), rule_6.to(device)

            outputs = model(kg_input=kg_input, input_ids=input_ids, attention_mask=attention_mask, rule_1=rule_1, rule_2=rule_2, rule_3=rule_3, rule_4=rule_4, rule_5=rule_5, rule_6=rule_6)
            logits = outputs.logits
            test_predictions.extend(logits.argmax(dim=1).tolist())
            test_true_labels.extend(batch_labels.tolist())

    test_accuracy = accuracy_score(test_true_labels, test_predictions)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(len(test_true_labels), len(test_predictions))
    print("!!!!!!\n", test_true_labels)
    print("\n!!!!!!\n", test_predictions)
