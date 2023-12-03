# -*- coding: utf-8 -*-
"""Entail_FR.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mcKHhRJoCHFN5ZLWVlfOEglr1dNJxBdc
"""

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Change the current working directory to a specific folder in your Google Drive
import os
os.chdir('/content/drive/My Drive/Colab/NLP Project/final/Datasets')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("french.csv")

# Split dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Label prompts
prompts = {
    0: "Ce texte ne contient aucune haine",
    1: "Ce texte contient de la haine"
}

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")

# Create text-prompt pairs
def get_input_pairs(texts, labels):
    input_texts = []
    for text, label in zip(texts, labels):
        text_pair = text + " [SEP] " + prompts[label]
        input_texts.append(text_pair)
    return input_texts

# Updated Dataset class
class HateDataset(Dataset):
    def __init__(self, texts, labels, languages, max_length=128):
        self.texts = texts
        self.labels = labels
        self.languages = languages
        self.max_length = max_length

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        lang = self.languages[idx]

        # Use the updated tokenizer
        encoding = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=self.max_length)

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': label}

# Create DataLoader
train_texts = get_input_pairs(train_df['text'].tolist(), train_df['label'].tolist())
test_texts = get_input_pairs(test_df['text'].tolist(), test_df['label'].tolist())

# Usage
train_dataset = HateDataset(train_texts, train_df['label'].tolist())
test_dataset = HateDataset(test_texts, test_df['label'].tolist())

# Rest of the code remains the same...
train_loader = DataLoader(train_dataset,batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8,shuffle=False)

# Model
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 3
loss_fn = torch.nn.CrossEntropyLoss()

train_losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_losses = []

    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        epoch_losses.append(loss.item())

        loss.backward()
        optimizer.step()

    average_loss = sum(epoch_losses) / len(epoch_losses)
    train_losses.append(average_loss)
    print(epoch, " done")

# Evaluation on Test Set
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc='Evaluating'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].tolist()

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).tolist()

        all_preds.extend(preds)
        all_labels.extend(labels)


# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='binary')

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"F1 Score: {f1*100:.2f}%")