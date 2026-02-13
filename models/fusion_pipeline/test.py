import os
import torch
import torch.nn as nn
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader

DATA_PATH = "../../data/TESS Toronto emotional speech set data"

texts = []
audio_features = []
labels = []

print("Loading multimodal test data...")

for folder in os.listdir(DATA_PATH):
    folder_path = os.path.join(DATA_PATH, folder)
    if not os.path.isdir(folder_path):
        continue

    emotion = folder.split("_")[-1].lower()

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)

            audio, sr = librosa.load(file_path, sr=16000)
            max_len = 3 * 16000
            if len(audio) > max_len:
                audio = audio[:max_len]
            else:
                audio = np.pad(audio, (0, max_len - len(audio)))

            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            audio_features.append(mfcc)

            word = file.replace(".wav", "").split("_")[1]
            texts.append(word)

            labels.append(emotion)

le = LabelEncoder()
labels = le.fit_transform(labels)

X_train_audio, X_test_audio, X_train_text, X_test_text, y_train, y_test = train_test_split(
    audio_features, texts, labels, test_size=0.2, random_state=42
)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class FusionDataset(Dataset):
    def __init__(self, audio, text, labels):
        self.audio = audio
        self.text = text
        self.labels = labels

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.text[idx],
            padding="max_length",
            truncation=True,
            max_length=10,
            return_tensors="pt"
        )

        return {
            "audio": torch.tensor(self.audio[idx], dtype=torch.float32),
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

test_dataset = FusionDataset(X_test_audio, X_test_text, y_test)
test_loader = DataLoader(test_dataset, batch_size=16)

class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super(FusionModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.lstm = nn.LSTM(
            input_size=32 * 20,
            hidden_size=64,
            batch_first=True,
            bidirectional=True
        )

        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.fc1 = nn.Linear(128 + 768, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, audio, input_ids, attention_mask):

        x = audio.unsqueeze(1)
        x = self.cnn(x)
        batch, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(batch, width, channels * height)
        x, _ = self.lstm(x)
        speech_embed = x[:, -1, :]

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_embed = outputs.pooler_output

        combined = torch.cat((speech_embed, text_embed), dim=1)
        x = self.fc1(combined)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FusionModel(num_classes=len(set(labels))).to(device)
model.load_state_dict(torch.load("fusion_model.pth"))
model.eval()

correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        audio = batch["audio"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        yb = batch["label"].to(device)

        outputs = model(audio, input_ids, attention_mask)
        _, predicted = torch.max(outputs, 1)

        correct += (predicted == yb).sum().item()
        total += yb.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

accuracy = correct / total

print("Fusion Test Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(all_labels, all_preds))
print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
