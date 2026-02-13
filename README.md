#  Multimodal Emotion Recognition

##  Overview

This project implements a **Multimodal Emotion Recognition System** using:

-  Speech-only input  
-  Text-only input  
-  Fusion of Speech + Text

The system is built using Deep Learning models to analyze emotional patterns from audio signals and textual content.

Dataset used: **Toronto Emotional Speech Set (**TESS**)** Available on Kaggle.

---

##  Objective

To build and compare three models:

## Speech-only emotion recognition

## Text-only emotion recognition ## Multimodal (Speech + Text) fusion model

And analyze:
- Which emotions are easiest/hardest to classify
- When fusion improves performance
- Cluster separability using t-**SNE** visualizations

---

##  Project Structure
``` bash
project/
│
├── models/
│   ├── speech_pipeline/
│   │   ├── train.py
│   │   └── test.py
│   │
│   ├── text_pipeline/
│   │   ├── train.py
│   │   └── test.py
│   │
│   └── fusion_pipeline/
│       ├── train.py
│       └── test.py
│
├── Results/
│   ├── accuracy_tables.csv
│   └── plots/
│       ├── accuracy_comparison.png
│       ├── tsne_speech.png
│       ├── tsne_text.png
│       └── tsne_fusion.png
│
├── Multimodal_Emotion_Recognition_Report.pdf
├── README.md
└── requirements.txt
```


---

## 🧠 Model Architectures

###  1. Speech Pipeline

**Preprocessing**
- Resampled to 16kHz
- Fixed length padding (3 seconds)
- **MFCC** feature extraction (40 coefficients)

**Architecture**
- **CNN** (local acoustic features)
- BiLSTM (temporal emotional patterns)
- Fully Connected + Softmax classifier

---

###  2. Text Pipeline

**Preprocessing**
- Extracted word from filename
- Tokenized using **BERT** tokenizer

**Architecture**
- **BERT** (bert-base-uncased)
- Dropout
- Fully Connected layer
- Softmax classifier

---

###  3. Fusion Pipeline

- Speech embedding (**CNN** + BiLSTM)
- Text embedding (**BERT**)
- Concatenation
- Fully Connected layers
- Softmax classifier

---

##  Experimental Results

| Model        | Test Accuracy |
|--------------|--------------|
| Speech-only  | 86.07% |
| Text-only    | 13.21% |
| Fusion       | 57.50% |

---

##  Visualization

### 🔹 Model Comparison

<img width="640" height="480" alt="accuracy_comparison" src="https://github.com/user-attachments/assets/57d6e9ac-45df-4c0d-a19c-dc5542ff0c76" />
`accuracy_comparison.png`


Speech > Fusion > Text

### 🔹 t-SNE Visualizations

<img width="640" height="480" alt="tsne_speech" src="https://github.com/user-attachments/assets/b9458751-03cc-48ca-ba7a-1f3e651d2696" />

- `tsne_speech.png` → Clear emotional clustering
<img width="640" height="480" alt="tsne_text" src="https://github.com/user-attachments/assets/c8fd0912-dce7-47e4-b9d3-b7877ef54bb4" />

- `tsne_text.png` → Poor separation (text lacks emotion)
<img width="640" height="480" alt="tsne_fusion" src="https://github.com/user-attachments/assets/ed9f9e29-74bb-4e9b-bf7e-5f8ca551eaf6" />

- `tsne_fusion.png` → Moderate clustering

---

##  Analysis

###  Easiest Emotions

- Happy
- Disgust
- Angry

Strong acoustic variations make them easier to classify.

###  Hardest Emotions

- Neutral
- Fear

Subtle acoustic differences lead to confusion.

###  When Does Fusion Help?

Fusion improves performance compared to text-only model. However, since **TESS** contains neutral spoken words, text adds limited emotional information, so fusion does not outperform speech-only model.

###  Error Analysis

## Fear misclassified as Angry due to similar high pitch.

## Neutral confused with Sad due to low energy overlap. ## Fusion model misclassified some classes because text introduced noise. ## Class imbalance influenced predictions in some cases.

---

##  Installation

Clone repository:

git clone [https://github.com/kattabhavana9/Multimodal-Emotion-Recognition.git](https://github.com/kattabhavana9/Multimodal-Emotion-Recognition.git) cd Multimodal-Emotion-Recognition

Install dependencies:

pip install -r requirements.txt

---

##  How to Run

###  Speech Model

cd models/speech_pipeline python train.py python test.py

###  Text Model

cd models/text_pipeline python train.py python test.py

###  Fusion Model

cd models/fusion_pipeline python train.py python test.py

---

##  Dependencies

- torch
- torchaudio
- librosa
- transformers
- scikit-learn
- matplotlib
- numpy
- tqdm

---

## 🚀 Author

**Katta Bhavana** Multimodal Emotion Recognition – Assignment 2
