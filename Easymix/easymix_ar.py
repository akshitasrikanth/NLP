
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import XLMRobertaTokenizer, XLMRobertaModel, AdamW
from sklearn.metrics import classification_report, accuracy_score, f1_score
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
df = pd.read_csv("arab.csv")

# Split the dataset into training and testing sets
# Here, we use a simple random split, you can use a more sophisticated split method
train_size = int(0.8 * len(df))
test_size = len(df) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(df, [train_size, test_size])

# Tokenizer and model
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
roberta_model = XLMRobertaModel.from_pretrained('xlm-roberta-base')

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

# Define the model with Easymix
class XLMRobertaEasymixModel(nn.Module):
    def __init__(self):
        super(XLMRobertaEasymixModel, self).__init__()
        self.roberta = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.classifier_head = ClassifierHead(self.roberta.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask)

        # Use the output of the classification head as logits
        logits = self.classifier_head(outputs.last_hidden_state.mean(dim=1))

        return logits

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['text'])
        label = int(self.data.iloc[idx]['label'])
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        label = torch.tensor(label)

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': label}

# Function to train the model
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

# Function to evaluate the model
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

# Function to train the XLM-RoBERTa Easymix model
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

# Function to evaluate the XLM-RoBERTa Easymix model
def evaluate1(model, dataloader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)

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

# DataLoader
train_dataloader = DataLoader(CustomDataset(train_dataset.dataset, tokenizer), batch_size=8, shuffle=True)
test_dataloader = DataLoader(CustomDataset(test_dataset.dataset, tokenizer), batch_size=8, shuffle=False)

# Training parameters
epochs = 5
learning_rate = 1e-5

# Initialize models
xlmroberta_model = XLMRobertaModel.from_pretrained('xlm-roberta-base')
classifier_head = ClassifierHead(xlmroberta_model.config.hidden_size, 2)

xlmroberta_easymix_model = XLMRobertaEasymixModel()

# Define optimizer and criterion
optimizer_xlmroberta = AdamW(xlmroberta_model.parameters(), lr=learning_rate)
optimizer_xlmroberta_easymix = AdamW(xlmroberta_easymix_model.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()

xmlroberta_model = xmlroberta_model.to('cuda')
optimizer_xmlroberta = AdamW(xmlroberta_model.parameters(), lr=learning_rate)

xmlroberta_easymix_model = xmlroberta_easymix_model.to('cuda')
optimizer_xmlroberta_easymix = AdamW(xmlroberta_easymix_model.parameters(), lr=learning_rate)

# Training and evaluation loops
xlmroberta_train_losses = []
xlmroberta_test_losses = []
xlmroberta_easymix_train_losses = []
xlmroberta_easymix_test_losses = []

# Train XLM-RoBERTa model
for epoch in range(epochs):
    xlmroberta_train_loss = train(xlmroberta_model, train_dataloader, optimizer_xlmroberta, criterion, device)
    xlmroberta_train_losses.append(xlmroberta_train_loss)

    xlmroberta_test_loss, xlmroberta_preds, xlmroberta_labels = evaluate(xlmroberta_model, test_dataloader, criterion, device)
    xlmroberta_test_losses.append(xlmroberta_test_loss)
    print(epoch," done")

# Train XML-RoBERTa easymix model
for epoch in range(epochs):
    xlmroberta_easymix_train_loss = train(xlmroberta_easymix_model, train_dataloader, optimizer_xlmroberta_easymix, criterion, device)
    xlmroberta_easymix_train_losses.append(xlmroberta_easymix_train_loss)

    xlmroberta_easymix_test_loss, xlmroberta_easymix_preds, xlmroberta_easymix_labels = evaluate(xlmroberta_easymix_model, test_dataloader, criterion, device)
    xlmroberta_easymix_test_losses.append(xlmroberta_easymix_test_loss)
    print(epoch," done")

# Evaluate and compare results
xlmroberta_accuracy = accuracy_score(xlmroberta_labels, xlmroberta_preds)
xlmroberta_f1 = f1_score(xlmroberta_labels, xlmroberta_preds)

xlmroberta_easymix_accuracy = accuracy_score(xlmroberta_easymix_labels, xlmroberta_easymix_preds)
xlmroberta_easymix_f1 = f1_score(xlmroberta_easymix_labels, xlmroberta_easymix_preds)

# Print and compare accuracy and f1-score
print("XLM-RoBERTa Model:")
print("Accuracy: {:.2%}".format(xlmroberta_accuracy))
print("F1-Score: {:.2%}".format(xlmroberta_f1))

print("\nXLM-RoBERTa+Easymix Model:")
print("Accuracy: {:.2%}".format(xlmroberta_easymix_accuracy))
print("F1-Score: {:.2%}".format(xlmroberta_easymix_f1))