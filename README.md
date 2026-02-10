# Text Analytics & NLP

**Graduate-level course materials for Text Mining & Social Media Analytics**  
Loyola Marymount University

---

## Overview

This repository contains Jupyter notebooks covering the complete text analytics pipeline—from raw text preprocessing to deep learning classification. Materials emphasize practical implementation with clear, pedagogical code designed for copy-paste learning.

---

## Course Structure

### 1. Text Preprocessing & Cleaning
| Notebook | Description |
|----------|-------------|
| `Text_Pre_Processing.ipynb` | Tokenization, stopword removal, stemming, lemmatization, regex cleaning |
| `Complete_NLP_Pipeline.ipynb` | End-to-end pipeline: raw text → clean text → features → model |

### 2. Feature Engineering & Text Representation
| Notebook | Description |
|----------|-------------|
| `Feature_Engineering_n_grams_tf_idf.ipynb` | Bag of Words, N-grams, TF-IDF vectorization |
| `Word2Vec.ipynb` | Word embeddings using Skip-gram and CBOW |
| `GloVe.ipynb` | Global Vectors for word representation |
| `Fasttext.ipynb` | Subword embeddings, OOV word handling |
| `Spacy_Token2Vec.ipynb` | spaCy's built-in word vectors |
| `Sentence2Vec_ipynb.ipynb` | Document-level embeddings via averaging and Sentence-BERT |

### 3. Text Classification (Traditional ML)
| Notebook | Description |
|----------|-------------|
| `binary_classification.ipynb` | Spam detection (2-class), precision/recall focus |
| `multiclass_classification.ipynb` | BBC news categorization (5-class), macro/weighted metrics |
| `multilabel_classification.ipynb` | Toxic comment detection (6 labels), Hamming loss, Jaccard score |
| `text_classification_binary_multiclass_multilabel_combined.ipynb` | All three classification types in one comprehensive notebook |

**Algorithms covered:** Naive Bayes, Logistic Regression, SVM, Random Forest

### 4. Deep Learning for Text
| Notebook | Description |
|----------|-------------|
| `Text_Classification_RNN_DL.ipynb` | Vanilla RNN for sequence classification |
| `Text_Classification_LSTM_BiLSTM_DL.ipynb` | LSTM and Bidirectional LSTM architectures |
| `Text_Classification_GRU_DL.ipynb` | GRU networks for text |

### 5. Unsupervised Methods
| Notebook | Description |
|----------|-------------|
| `TopicModeling_LDA_LSA_NMF_.ipynb` | Latent Dirichlet Allocation, LSA, Non-negative Matrix Factorization |
| `Sentiment_Analysis_Vader_TextBlob.ipynb` | Rule-based sentiment with VADER and TextBlob |

---

## Data

The `Data/` folder contains datasets used across notebooks:
- **Email spam** — Binary classification
- **BBC News** — Multiclass (5 categories: business, entertainment, politics, sport, tech)
- **Toxic Comments** — Multilabel (toxic, severe_toxic, obscene, threat, insult, identity_hate)

---

## Key Concepts Covered

- **Pipeline stages:** Text Cleaning → Preprocessing → Feature Engineering → Modeling → Evaluation
- **Embeddings:** Static (Word2Vec, GloVe, FastText) vs. Contextual (BERT)
- **Evaluation:** Confusion matrix, precision/recall by class, F1-score, class imbalance handling
- **The 80/20 rule:** Data preparation is ~80% of NLP work

---

## Requirements

```
numpy
pandas
scikit-learn
nltk
spacy
gensim
tensorflow / keras
transformers
sentence-transformers
```

---

## Usage

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download NLTK data: `nltk.download('punkt')`, `nltk.download('stopwords')`, etc.
4. Download spaCy model: `python -m spacy download en_core_web_sm`
5. Run notebooks in order by topic

---

## Author

Ace Vo  
Loyola Marymount University

---

## License

Educational use. Materials designed for graduate-level instruction.
