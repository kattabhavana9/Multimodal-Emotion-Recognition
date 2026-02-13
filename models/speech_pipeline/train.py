import os
import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
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

print("Loading data...")
from sklearn.model_selection import train_test_split

X, y = load_data()

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Total samples loaded:", len(X))


le = LabelEncoder()
y = le.fit_transform(y)

X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train, dtype=torch.long)


dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

class SpeechModel(nn.Module):
    def __init__(self, num_classes):
        super(SpeechModel, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.lstm = nn.LSTM(
            input_size=32 * 20,   # IMPORTANT
            hidden_size=64,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(1)   # (batch, 1, 40, time)
        x = self.cnn(x)      # (batch, 32, 20, time/2)

        # reshape for LSTM
        batch, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2)   # (batch, time, channels, height)
        x = x.contiguous().view(batch, width, channels * height)

        x, _ = self.lstm(x)

        x = x[:, -1, :]
        x = self.fc(x)

        return x


model = SpeechModel(num_classes=len(set(y)))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training started...")

for epoch in range(10):
    total_loss = 0
    correct = 0
    total = 0

    for xb, yb in loader:
        outputs = model(xb)
        loss = criterion(outputs, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == yb).sum().item()
        total += yb.size(0)

    accuracy = correct / total
    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Accuracy: {accuracy:.4f}")

print("Training complete!")
torch.save(model.state_dict(), "speech_model.pth")
print("Model saved successfully!")
