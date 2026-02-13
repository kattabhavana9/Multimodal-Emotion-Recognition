import os
import torch
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = "../data/TESS Toronto emotional speech set data"

texts = []
labels = []

print("Extracting text embeddings...")

for folder in os.listdir(DATA_PATH):
    folder_path = os.path.join(DATA_PATH, folder)
    if not os.path.isdir(folder_path):
        continue

    emotion = folder.split("_")[-1].lower()

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            word = file.replace(".wav", "").split("_")[1]
            texts.append(word)
            labels.append(emotion)

le = LabelEncoder()
labels = le.fit_transform(labels)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

embeddings = []

with torch.no_grad():
    for text in texts[:1000]:
        encoding = tokenizer(text, return_tensors="pt")
        outputs = model(**encoding)
        embeddings.append(outputs.pooler_output.squeeze().numpy())

embeddings = np.array(embeddings)
labels = labels[:1000]

tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(embeddings)

plt.figure()
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels)
plt.title("t-SNE of Text Embeddings")
plt.colorbar(scatter)
plt.savefig("tsne_text.png")
plt.show()
