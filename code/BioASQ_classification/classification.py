import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, AutoTokenizer, AutoModelForSequenceClassification, RobertaForSequenceClassification, RobertaTokenizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score


##data preprocessing
with open('../Data/bioasq/bioasq_10b_yesno.json','r') as f: ##bioasq_10b_yesno.json
    data= json.load(f)


train_data=data[:950]
val_data=data[950:975]
test_data=data[975:]
#### biobert best case
# train_data=data[:750]
# val_data=data[750:950]
# test_data=data[950:]

# Define a function for preprocessing the data
def preprocess_data(data):
    contexts = []
    questions = []
    labels = []

    for item in data:
        context = item['context']
        question = item['question']
        label = item['exact_answer']  # 'yes' or 'no'

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

    # Determine the maximum sequence length among all examples
    max_seq_length = max(len(seq) for seq in contexts)

    # Pad sequences to the maximum length
    contexts = [torch.cat([seq, torch.zeros(max_seq_length - len(seq)).long()]) for seq in contexts]
    questions = [torch.cat([seq, torch.zeros(max_seq_length - len(seq)).long()]) for seq in questions]

    return torch.stack(contexts), torch.stack(questions), torch.tensor(labels)


##### Load the tokenizer
# tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
tokenizer = AutoTokenizer.from_pretrained('michiyasunaga/BioLinkBERT-large')
# tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

# Preprocess the training data
train_contexts, train_questions, train_labels = preprocess_data(train_data)

# Create PyTorch DataLoader for training data
train_dataset = TensorDataset(train_contexts, train_questions, train_labels)
batch_size = 2  # Adjust as needed
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# Similarly, preprocess the validation and test data
val_contexts, val_questions, val_labels = preprocess_data(val_data)
test_contexts, test_questions, test_labels = preprocess_data(test_data)

val_dataset = TensorDataset(val_contexts, val_questions, val_labels)
test_dataset = TensorDataset(test_contexts, test_questions, test_labels)

val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle validation
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle test

# Initialize the BioBERT model for sequence classification
# model = BertForSequenceClassification.from_pretrained("dmis-lab/biobert-v1.1", num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained('michiyasunaga/BioLinkBERT-large', num_labels=2)
# model = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=2)


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
        input_ids, attention_mask, batch_labels = batch
        input_ids, attention_mask, batch_labels = input_ids.to(device), attention_mask.to(device), batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=batch_labels)
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
            input_ids, attention_mask, batch_labels = batch
            input_ids, attention_mask, batch_labels = input_ids.to(device), attention_mask.to(device), batch_labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
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
            input_ids, attention_mask, batch_labels = batch
            input_ids, attention_mask, batch_labels = input_ids.to(device), attention_mask.to(device), batch_labels.to(device)

            outputs = best_model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            test_predictions.extend(logits.argmax(dim=1).tolist())
            test_true_labels.extend(batch_labels.tolist())

    test_accuracy = accuracy_score(test_true_labels, test_predictions)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(len(test_true_labels), len(test_predictions))
    print("!!!!!!\n", test_true_labels)
    print("\n!!!!!!\n", test_predictions)