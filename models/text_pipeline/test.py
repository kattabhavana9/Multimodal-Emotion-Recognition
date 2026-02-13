import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_PATH = "../../data/TESS Toronto emotional speech set data"

texts = []
labels = []

print("Loading text data...")

for folder in os.listdir(DATA_PATH):
    folder_path = os.path.join(DATA_PATH, folder)
    if not os.path.isdir(folder_path):
        continue

    emotion = folder.split("_")[-1].lower()

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            parts = file.replace(".wav", "").split("_")
            word = parts[1]
            texts.append(word)
            labels.append(emotion)

le = LabelEncoder()
labels = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=10,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

test_dataset = TextDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=16)

class TextModel(nn.Module):
    def __init__(self, num_classes):
        super(TextModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        x = self.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextModel(num_classes=len(set(labels))).to(device)
model.load_state_dict(torch.load("text_model.pth"))
model.eval()

correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        yb = batch["label"].to(device)

        outputs = model(input_ids, attention_mask)
        _, predicted = torch.max(outputs, 1)

        correct += (predicted == yb).sum().item()
        total += yb.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

accuracy = correct / total

print("Text Test Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(all_labels, all_preds))
print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
