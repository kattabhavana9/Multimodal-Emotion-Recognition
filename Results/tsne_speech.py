import torch
import torch.nn as nn
import librosa
import os
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DATA_PATH = "../data/TESS Toronto emotional speech set data"

audio_features = []
labels = []

print("Extracting speech embeddings...")

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
            audio_features.append(mfcc.flatten())
            labels.append(emotion)

le = LabelEncoder()
labels = le.fit_transform(labels)

X = np.array(audio_features)

# Reduce size for speed
X = X[:1000]
labels = labels[:1000]

tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X)

plt.figure()
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels)
plt.title("t-SNE of Speech Features")
plt.colorbar(scatter)

plt.savefig("tsne_speech.png")
plt.show()
