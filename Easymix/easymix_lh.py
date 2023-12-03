
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel, AdamW
from sklearn.metrics import classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd

# Set a random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Load the dataset (replace this with your actual dataset loading logic)
# Assume you have a CSV file with columns: 'text' and 'label'
df = pd.read_csv("cleaned_latenthatred.csv")

# Split the dataset into training and testing sets
# Here, we use a simple random split, you can use a more sophisticated split method
train_size = int(0.8 * len(df))
test_size = len(df) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(df, [train_size, test_size])

# Tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaModel.from_pretrained('roberta-base')

len(df)

len(train_dataset)

len(test_dataset)

# Define a simple classification head
class ClassifierHead(nn.Module):
    def __init__(self, input_size, output_size):
        super(ClassifierHead, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)

            outputs = model(input_ids, attention_mask)

            # Extract logits from pooler_output
            logits = outputs['pooler_output']

            # Apply softmax to logits
            logits = F.softmax(logits, dim=1)

            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_preds, all_labels

# Define the model with Easymix



class RobertaEasymixModel(nn.Module):
    def __init__(self):
        super(RobertaEasymixModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.classifier_head = ClassifierHead(self.roberta.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask)

        # Use the output of the classification head as logits
        logits = self.classifier_head(outputs['pooler_output'])

        return logits

import torch.nn.functional as F

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)

        # Extract logits from pooler_output
        logits = outputs['pooler_output']

        # Apply softmax to logits
        logits = F.softmax(logits, dim=1)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        text = str(self.data.iloc[idx]['text'])
        label = int(self.data.iloc[idx]['label'])
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

        # Move tensors to the same device
        input_ids = encoding['input_ids'].squeeze().to('cuda')
        attention_mask = encoding['attention_mask'].squeeze().to('cuda')
        label = torch.tensor(label).to('cuda')

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': label}

train_dataloader = DataLoader(CustomDataset(train_dataset.dataset, tokenizer), batch_size=8, shuffle=True)
test_dataloader = DataLoader(CustomDataset(test_dataset.dataset, tokenizer), batch_size=8, shuffle=False)

# Training parameters
epochs = 5
learning_rate = 1e-5

# Initialize models
roberta_model = RobertaModel.from_pretrained('roberta-base')
classifier_head = ClassifierHead(roberta_model.config.hidden_size, 2)

roberta_easymix_model = RobertaEasymixModel()

# Define optimizer and criterion
optimizer_roberta = AdamW(roberta_model.parameters(), lr=learning_rate)
optimizer_roberta_easymix = AdamW(roberta_easymix_model.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()

roberta_model = roberta_model.to('cuda')
optimizer_roberta = AdamW(roberta_model.parameters(), lr=learning_rate)

roberta_easymix_model = roberta_easymix_model.to('cuda')
optimizer_roberta_easymix = AdamW(roberta_easymix_model.parameters(), lr=learning_rate)

def train1(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids, attention_mask, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate1(model, dataloader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['label'].to(device)

            outputs = model(input_ids, attention_mask)

            # Assuming the output is a tensor, not a dictionary
            logits = outputs  # Adjust this line based on the actual structure of the 'outputs' object

            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_preds, all_labels

#Training and evaluation loops
roberta_train_losses = []
roberta_test_losses = []
roberta_easymix_train_losses = []
roberta_easymix_test_losses = []

# Train Roberta model

for epoch in range(epochs):

    roberta_train_loss = train(roberta_model, train_dataloader, optimizer_roberta, criterion, 'cuda')
    roberta_train_losses.append(roberta_train_loss)

    roberta_test_loss, roberta_preds, roberta_labels = evaluate(roberta_model, test_dataloader, criterion, 'cuda')
    roberta_test_losses.append(roberta_test_loss)
    print(epoch," done")

# Train Roberta easymix model
train_dataloader = [(d['input_ids'].to('cuda'), d['attention_mask'].to('cuda'), d['label'].to('cuda')) for d in train_dataloader]

for epoch in range(epochs):
    # Train Roberta with Easymix
    roberta_easymix_train_loss = train1(roberta_easymix_model, train_dataloader, optimizer_roberta_easymix, criterion, 'cuda')
    roberta_easymix_train_losses.append(roberta_easymix_train_loss)

    roberta_easymix_test_loss, roberta_easymix_preds, roberta_easymix_labels = evaluate(roberta_easymix_model, test_dataloader, criterion, 'cuda')
    roberta_easymix_test_losses.append(roberta_easymix_test_loss)
    print(epoch," done")

# Evaluate and compare results
roberta_accuracy = accuracy_score(roberta_labels, roberta_preds)
roberta_f1 = f1_score(roberta_labels, roberta_preds)

roberta_easymix_accuracy = accuracy_score(roberta_easymix_labels, roberta_easymix_preds)
roberta_easymix_f1 = f1_score(roberta_easymix_labels, roberta_easymix_preds)


# Print and compare accuracy and f1-score
print("RoBERTa Model:")
print("Accuracy: {:.2%}".format(roberta_accuracy))
print("F1-Score: {:.2%}".format(roberta_f1))

print("\nRoBERTa+Easymix Model:")
print("Accuracy: {:.2%}".format(roberta_easymix_accuracy))
print("F1-Score: {:.2%}".format(roberta_easymix_f1))