import torch
import torch.nn as nn
import numpy as np
import os
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

DATA_PATH = "../../data/TESS Toronto emotional speech set data"

def load_data():
    X = []
    y = []
    
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
                X.append(mfcc)
                y.append(emotion)
    
    return np.array(X), np.array(y)

class SpeechModel(nn.Module):
    def __init__(self, num_classes):
        super(SpeechModel, self).__init__()
        
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
        
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        batch, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(batch, width, channels * height)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

print("Loading test data...")
from sklearn.model_selection import train_test_split

X, y = load_data()

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


X_tensor = torch.tensor(X_test, dtype=torch.float32)
y_tensor = torch.tensor(y_test, dtype=torch.long)


dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32)

model = SpeechModel(num_classes=len(set(y)))
model.load_state_dict(torch.load("speech_model.pth"))
model.eval()

correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for xb, yb in loader:
        outputs = model(xb)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == yb).sum().item()
        total += yb.size(0)
        all_preds.extend(predicted.numpy())
        all_labels.extend(yb.numpy())

accuracy = correct / total
print("Test Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(all_labels, all_preds))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
