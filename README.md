# MedEx: Enhancing Medical Question-Answering with First-Order Logic based Reasoning and Knowledge Injection

This README file provides step-by-step instructions for reproducing the MedEx experiment. It covers data creation, training, and inference processes.
## Data Creation

### 1. KG Construction Using QUICK-UMLS
To convert the dataset into required format, follow these steps:
1. Run the script located at `Data/preprocess_kg/final_preprocess.py`.
2. Save the output file as `mashqa_train_data.json`.


### 2. Integrate RULE with KG Data
To integrate rules with the MashQA KG train data, follow these steps:
1. Run the script located at `Data/Rule_integrate/cosine_triple.py`. This will save the file as `mashqa_data_withRule`.
2. Run the script `Data/data_preparation.py` to create a file named `MashQA_train_data_with_rule.json`.

### 3. Data preparation
1. Run the script located at `code/ExtractiveQA/preprocess/fol_preprocess.py`.
2. Run the script located at `code/ExtractiveQA/preprocess/data_preprocess.py`.

#### Training
1. Run the script located at `code/ExtractiveQA/qa_st.py`.
2. The script will read the file `Data/MashQA_train_data_with_rule.json` prepared earlier.


---
This README file aims to provide clear and comprehensive instructions to facilitate the replication of the MedEx experiment.
