# Sarcasm Detection Project Summary

## Project Overview
This project implements and compares multiple deep learning approaches for sarcasm detection in news headlines using the Sarcasm Headlines Dataset v2.

---

## Methods Implemented

### 1. Word2Vec with Bidirectional LSTM-GRU
**Embedding Approach:** Static word embeddings using Word2Vec
- **Embedding Dimension:** 200
- **Training Method:** Word2Vec trained on the corpus with window size 5, min_count 1
- **Model Architecture:**
  - Embedding layer (trainable, initialized with Word2Vec vectors)
  - Bidirectional LSTM (128 units, 30% dropout, 30% recurrent dropout)
  - Bidirectional GRU (32 units, 10% dropout, 10% recurrent dropout)
  - Dense output layer with sigmoid activation
- **Training Configuration:**
  - Optimizer: Adam (learning_rate=0.01)
  - Loss: Binary crossentropy
  - Metrics: F1 Score
  - Epochs: 5
  - Batch size: 128
  - Sequence length: 20
  - Train/Test split: 70/30

**Key Features:**
- Custom-trained embeddings on the sarcasm dataset
- Bidirectional processing for context from both directions
- Combined LSTM and GRU layers for hierarchical feature extraction

---

### 2. GloVe with Bidirectional LSTM
**Embedding Approach:** Pre-trained GloVe embeddings (Twitter 27B, 25d)
- **Embedding Dimension:** 25
- **Embedding Source:** Pre-trained GloVe Twitter embeddings
- **Model Architecture:**
  - Embedding layer (trainable, initialized with GloVe vectors)
  - Bidirectional LSTM (128 units, 50% dropout, 50% recurrent dropout)
  - Dense output layer with sigmoid activation
- **Training Configuration:**
  - Optimizer: Adam (learning_rate=0.01)
  - Loss: Binary crossentropy
  - Metrics: Accuracy
  - Epochs: 2
  - Batch size: 128
  - Sequence length: 200
  - Train/Test split: 70/30

**Key Features:**
- Pre-trained embeddings capture general Twitter language patterns
- Longer sequence length (200 tokens) for more context
- Higher dropout rates (50%) to prevent overfitting

---

### 3. BERT Fine-tuning (Transformer-based LLM)
**Model:** BERT-base-uncased (Transformer encoder)
- **Architecture:** 12 transformer encoder layers, 768 hidden dim, 12 attention heads
- **Parameters:** ~110 million (fine-tuned for binary classification)
- **Tokenization:** BERT WordPiece tokenizer with max length 128
- **Model Architecture:**
  - Pre-trained BERT encoder
  - Classification head (2 output classes)
- **Training Configuration:**
  - Optimizer: AdamW (learning_rate=2e-5)
  - Loss: Cross-entropy
  - Metrics: Accuracy, Precision, Recall, F1-score
  - Epochs: 5
  - Batch size: 32 (per device)
  - Weight decay: 0.01
  - FP16 mixed precision training
  - Train/Test split: 70/30
  - Evaluation strategy: Per epoch
  - Best model selection: Based on F1-score

**Key Features:**
- Bidirectional context understanding through transformer self-attention
- Transfer learning from large-scale pre-training
- Fine-tuned specifically for sarcasm detection task
- State-of-the-art contextual embeddings

---

## Data Preprocessing

### Common Preprocessing Steps:
1. **HTML Removal:** Strip HTML tags using BeautifulSoup
2. **Bracket Content Removal:** Remove text within square brackets
3. **URL Removal:** Remove HTTP/HTTPS URLs
4. **Stopword Removal:** Remove English stopwords and punctuation
5. **Text Cleaning:** Apply combined denoising function

### Dataset:
- **Source:** Sarcasm Headlines Dataset v2 (JSON format)
- **Features Used:** Headlines text
- **Target Variable:** Binary classification (sarcastic vs. non-sarcastic)
- **Split Ratio:** 70% training, 30% testing (random_state=0 for reproducibility)

---

## Model Comparison Summary

| Method | Embedding Type | Model Type | Parameters | Sequence Length | Epochs |
|--------|---------------|------------|------------|-----------------|--------|
| Word2Vec + LSTM-GRU | Static (custom) | RNN | ~Few million | 20 | 5 |
| GloVe + LSTM | Static (pre-trained) | RNN | ~Few million | 200 | 2 |
| BERT Fine-tuning | Contextual (transformer) | Transformer | ~110M | 128 | 5 |

---

## Technical Implementation Details

### Libraries & Frameworks:
- **Deep Learning:** TensorFlow/Keras, PyTorch (via Transformers)
- **NLP Processing:** NLTK, Hugging Face Transformers, Datasets
- **Embeddings:** Gensim (Word2Vec), GloVe pre-trained vectors
- **Data Processing:** Pandas, NumPy
- **Evaluation:** Scikit-learn metrics
- **Acceleration:** Hugging Face Accelerate (FP16 training)

### Key Techniques:
1. **Bidirectional Processing:** All models use bidirectional architectures to capture context from both directions
2. **Regularization:** Dropout and recurrent dropout to prevent overfitting
3. **Transfer Learning:** GloVe and BERT leverage pre-trained knowledge
4. **Mixed Precision:** FP16 training for BERT to improve speed and memory efficiency
5. **Early Stopping:** Best model selection based on validation F1-score

---

## Evaluation Metrics
All models evaluated using:
- **Accuracy:** Overall classification correctness
- **Precision:** Ratio of true positives to predicted positives
- **Recall:** Ratio of true positives to actual positives
- **F1-Score:** Harmonic mean of precision and recall
- **Classification Report:** Per-class performance breakdown

---

## Project Structure
```
Project3/
├── data/
│   ├── Sarcasm_Headlines_Dataset_v2.json
│   └── glove.twitter.27B.25d.txt
├── results/          # BERT training outputs
├── logs/            # Training logs
└── project3.ipynb   # Main implementation notebook
```
