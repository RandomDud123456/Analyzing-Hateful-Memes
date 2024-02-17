# import os
# import re
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# from transformers import BertTokenizer, BertModel
# from torch.optim import AdamW  
# from sklearn.model_selection import train_test_split
# from bs4 import BeautifulSoup
# from functools import partial
# from transformers import BertForSequenceClassification

# # Define the model
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# # Save the model
# torch.save(model, "bert_hate_classification_model.pth")
# print("Model saved successfully.")

# # Load the model
# model = torch.load("bert_hate_classification_model.pth")

# # Define other functions and variables
# num_correct = 0
# num_images = 0

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# def text_preprocessing(final):
#     preprocessed_text = []
#     for sentence in final:
#         sentence = BeautifulSoup(sentence, 'lxml').get_text()
#         sentence = decontracted(sentence)
#         sentence = re.sub("\S*\d\S*", "", sentence).strip()
#         sentence = re.sub('[^A-Za-z]+', ' ', sentence)
#         preprocessed_text.append(sentence.strip())
#     return preprocessed_text

# def decontracted(phrase):
#     # specific
#     phrase = re.sub(r"won't", "will not", phrase)
#     phrase = re.sub(r"can\'t", "can not", phrase)

#     # general
#     phrase = re.sub(r"n\'t", " not", phrase)
#     phrase = re.sub(r"\'re", " are", phrase)
#     phrase = re.sub(r"\'s", " is", phrase)
#     phrase = re.sub(r"\'d", " would", phrase)
#     phrase = re.sub(r"\'ll", " will", phrase)
#     phrase = re.sub(r"\'t", " not", phrase)
#     phrase = re.sub(r"\'ve", " have", phrase)
#     phrase = re.sub(r"\'m", " am", phrase)
#     return phrase

# def predict_text_hatefulness(model, tokenizer, text_file_path):
#     # Read text from file
#     with open(text_file_path, 'r', encoding='utf-8') as file:
#         text = file.read()

#     # Preprocess the text
#     text = text_preprocessing([text])[0]

#     # Tokenize the text
#     inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding=True)

#     # Perform inference
#     with torch.no_grad():
#         input_ids = inputs['input_ids']
#         attention_mask = inputs['attention_mask']
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         logits = outputs.logits
#         probabilities = nn.functional.softmax(logits, dim=1)
#         predicted_class = torch.argmax(probabilities, dim=1).item()

#     return predicted_class, probabilities[0][predicted_class].item()

# directory = "/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/dataset_hate/test/texts/1"

# # Initialize counters
# num_correct = 0
# num_images = 0

# # Iterate over the text files in the directory
# for filename in os.listdir(directory):
#     if filename.endswith(".txt"):
#         file_path = os.path.join(directory, filename)
        
#         # Predict hateful or not hateful
#         predicted_class, _ = predict_text_hatefulness(model, tokenizer, file_path)
        
#         # Increment num_images
#         num_images += 1
        
#         # If predicted class is hateful, increment num_correct
#         if predicted_class == 1:
#             num_correct += 1

# # Output the results
# print(f"Number of correct predictions: {num_correct}")
# print(f"Total number of images: {num_images}")

# import os
# import re
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# from transformers import BertTokenizer, BertModel
# from torch.optim import AdamW  
# from sklearn.model_selection import train_test_split
# from bs4 import BeautifulSoup
# from functools import partial
# from transformers import BertForSequenceClassification


# SLANG_PATH = "static/slang.txt"

# # Define a custom dataset class
# class CustomDataset(Dataset):
#     def __init__(self, root_dir, tokenizer, max_len):
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#         self.samples = []

#         for label in os.listdir(root_dir):
#             label_dir = os.path.join(root_dir, label)
#             for filename in os.listdir(label_dir):
#                 with open(os.path.join(label_dir, filename), 'r', encoding='utf-8') as file:
#                     text = file.read()
#                     text = text_preprocessing([text])[0]  # Apply text preprocessing
#                 self.samples.append((text, int(label)))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         text, label = self.samples[idx]

#         encoding = self.tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             max_length=self.max_len,
#             return_token_type_ids=False,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt'
#         )

#         return {
#             'text': text,
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'label': torch.tensor(label, dtype=torch.long)
#         }

# # Define text preprocessing functions
# def decontracted(phrase):
#     # specific
#     phrase = re.sub(r"won't", "will not", phrase)
#     phrase = re.sub(r"can\'t", "can not", phrase)

#     # general
#     phrase = re.sub(r"n\'t", " not", phrase)
#     phrase = re.sub(r"\'re", " are", phrase)
#     phrase = re.sub(r"\'s", " is", phrase)
#     phrase = re.sub(r"\'d", " would", phrase)
#     phrase = re.sub(r"\'ll", " will", phrase)
#     phrase = re.sub(r"\'t", " not", phrase)
#     phrase = re.sub(r"\'ve", " have", phrase)
#     phrase = re.sub(r"\'m", " am", phrase)
#     return phrase

# with open(SLANG_PATH) as file:
#     slang_map = dict(map(str.strip, line.partition('\t')[::2])
#                      for line in file if line.strip())

# slang_words = sorted(slang_map, key=len, reverse=True)
# regex = re.compile(r"\b({})\b".format("|".join(map(re.escape, slang_words))))
# replaceSlang = partial(regex.sub, lambda m: slang_map[m.group(1)])

# def text_preprocessing(final):
#     preprocessed_text = []
#     for sentence in final:
#         sentence = BeautifulSoup(sentence, 'lxml').get_text()
#         sentence = replaceSlang(sentence)
#         sentence = decontracted(sentence)
#         sentence = re.sub("\S*\d\S*", "", sentence).strip()
#         sentence = re.sub('[^A-Za-z]+', ' ', sentence)
#         preprocessed_text.append(sentence.strip())
#     return preprocessed_text

# # Load the dataset
# root_dir = "/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/dataset_hate/train/texts"
# test_root_dir="/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/dataset_hate/test/texts"

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# dataset = CustomDataset(root_dir, tokenizer, max_len=128)
# dataset_test=CustomDataset(test_root_dir, tokenizer, max_len=128)

# # Split dataset into training and testing sets
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# # Set up model
# class BertForSequenceClassification(nn.Module):
#     def __init__(self):
#         super(BertForSequenceClassification, self).__init__()
#         self.bert = BertModel.from_pretrained("bert-base-uncased")
#         self.dropout = nn.Dropout(0.1)
#         self.classifier = nn.Linear(768, 2)  # 2 output classes

#     def forward(self, input_ids, attention_mask, token_type_ids=None):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         pooled_output = outputs.pooler_output
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#         return logits

# model = BertForSequenceClassification()

# # Custom training parameters
# batch_size = 16
# epochs = 1
# learning_rate = 2e-5

# # Create dataloaders for training and testing sets
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# final_test_loader=DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = AdamW(model.parameters(), lr=learning_rate)  # Use AdamW from torch.optim

# # Training loop
# model.train()
# for epoch in range(epochs):
#     total_loss = 0
#     for batch in train_loader:
#         optimizer.zero_grad()
#         input_ids = batch['input_ids']
#         attention_mask = batch['attention_mask']
#         labels = batch['label']
#         outputs = model(input_ids, attention_mask=attention_mask)
#         loss = criterion(outputs, labels)
#         total_loss += loss.item()
#         loss.backward()
#         optimizer.step()
#     print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss}')

# # Evaluation loop
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for batch in test_loader:
#         input_ids = batch['input_ids']
#         attention_mask = batch['attention_mask']
#         labels = batch['label']
#         outputs = model(input_ids, attention_mask=attention_mask)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# accuracy = correct / total
# print(f'Accuracy on cross validation set: {accuracy}')

# # Test loop
# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for batch in final_test_loader:
#         input_ids = batch['input_ids']
#         attention_mask = batch['attention_mask']
#         labels = batch['label']
#         outputs = model(input_ids, attention_mask=attention_mask)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# accuracy = correct / total
# print(f'Accuracy on test set: {accuracy}')

import os
import re
from functools import partial
from bs4 import BeautifulSoup
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW
import torch
import torch.nn as nn

class CustomDataset(Dataset):
    def __init__(self, root_dir, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            for filename in os.listdir(label_dir):
                with open(os.path.join(label_dir, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                    text = text_preprocessing([text])[0]  # Apply text preprocessing
                self.samples.append((text, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Define text preprocessing functions
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

# Modify this path accordingly
SLANG_PATH = "static/slang.txt"

with open(SLANG_PATH) as file:
    slang_map = dict(map(str.strip, line.partition('\t')[::2])
                     for line in file if line.strip())

slang_words = sorted(slang_map, key=len, reverse=True)
regex = re.compile(r"\b({})\b".format("|".join(map(re.escape, slang_words))))
replaceSlang = partial(regex.sub, lambda m: slang_map[m.group(1)])

def text_preprocessing(final):
    preprocessed_text = []
    for sentence in final:
        sentence = BeautifulSoup(sentence, 'lxml').get_text()
        sentence = replaceSlang(sentence)
        sentence = decontracted(sentence)
        sentence = re.sub("\S*\d\S*", "", sentence).strip()
        sentence = re.sub('[^A-Za-z]+', ' ', sentence)
        preprocessed_text.append(sentence.strip())
    return preprocessed_text

# Load the dataset
root_dir = "/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/dataset_hate/train/texts"
test_root_dir="/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/dataset_hate/test/texts"

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = CustomDataset(root_dir, tokenizer, max_len=128)
dataset_test=CustomDataset(test_root_dir, tokenizer, max_len=128)

# Split dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Set up model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Custom training parameters
batch_size = 16
epochs = 3
learning_rate = 2e-5

# Create dataloaders for training and testing sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

final_test_loader=DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)  # Use AdamW from torch.optim

# Training loop
for epoch in range(epochs):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)  # Fixing the error here
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss}')

# Evaluation loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy on cross validation set: {accuracy}')

# Test loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in final_test_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy on test set: {accuracy}')


torch.save(model, "bert_hate_classification_model.pth")
print("Model saved successfully.")

def predict_text_hatefulness(model, tokenizer, text_file_path):
    # Read text from file
    with open(text_file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Preprocess the text
    text = text_preprocessing([text])[0]

    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding=True)

    # Perform inference
    with torch.no_grad():
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class, probabilities[0][predicted_class].item()

text_file_path = "/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/dataset_hate/test/texts/1/04127.txt"

# Call the predict_text_hatefulness function
predicted_class, confidence = predict_text_hatefulness(model, tokenizer, text_file_path)

# Interpret the prediction
class_names = ['Not Hateful', 'Hateful']
print(f'Text is {class_names[predicted_class]} with confidence {confidence:.2f}')

model = torch.load("bert_hate_classification_model.pth")

text_file_path = "/home/lenovo/Documents/Computer Vision/streamlit-hatefulmemedection-main/dataset_hate/test/texts/1/04127.txt"

# Call the predict_text_hatefulness function
predicted_class, confidence = predict_text_hatefulness(model, tokenizer, text_file_path)

# Interpret the prediction
class_names = ['Not Hateful', 'Hateful']
print(f'Text is {class_names[predicted_class]} with confidence {confidence:.2f}')