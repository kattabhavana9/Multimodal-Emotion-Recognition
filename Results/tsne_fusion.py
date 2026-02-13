import os
import torch
import torch.nn as nn
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel

DATA_PATH = "../data/TESS Toronto emotional speech set data"

texts = []
audio_features = []
labels = []

print("Extracting fusion embeddings...")

for folder in os.listdir(DATA_PATH):
    folder_path = os.path.join(DATA_PATH, folder)
    if not os.path.isdir(folder_path):
        continue

    emotion = folder.split("_")[-1].lower()

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):

            file_path = os.path.join(folder_path, file)

            # --- Speech ---
            audio, sr = librosa.load(file_path, sr=16000)
            max_len = 3 * 16000
            if len(audio) > max_len:
                audio = audio[:max_len]
            else:
                audio = np.pad(audio, (0, max_len - len(audio)))

            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            audio_features.append(mfcc.flatten())

            # --- Text ---
            word = file.replace(".wav", "").split("_")[1]
            texts.append(word)

            labels.append(emotion)

le = LabelEncoder()
labels = le.fit_transform(labels)

# Reduce size for speed
audio_features = audio_features[:1000]
texts = texts[:1000]
labels = labels[:1000]

# --- Text embeddings ---
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")
bert.eval()

text_embeddings = []

with torch.no_grad():
    for text in texts:
        encoding = tokenizer(text, return_tensors="pt")
        outputs = bert(**encoding)
        text_embeddings.append(outputs.pooler_output.squeeze().numpy())

text_embeddings = np.array(text_embeddings)

# --- Combine speech + text ---
audio_features = np.array(audio_features)

fusion_embeddings = np.concatenate((audio_features, text_embeddings), axis=1)

print("Running t-SNE...")

tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(fusion_embeddings)

plt.figure()
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels)
plt.title("t-SNE of Fusion Representations")
plt.colorbar(scatter)

plt.savefig("tsne_fusion.png")
plt.show()

print("Fusion t-SNE saved!")
