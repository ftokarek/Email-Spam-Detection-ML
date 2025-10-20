# Email Spam Detection

A machine learning project for detecting spam emails using LSTM neural networks with TensorFlow and Keras.

## Overview

This project implements a binary classification model to distinguish between spam and legitimate emails. The model uses natural language processing techniques and a Long Short-Term Memory (LSTM) architecture to achieve high accuracy in spam detection.

## Features

- Data preprocessing with stopword removal and punctuation cleaning
- Balanced dataset handling to prevent bias
- LSTM-based deep learning model
- Comprehensive evaluation metrics including confusion matrix and classification report
- Model persistence for future predictions

## Dataset

The project uses the spam_ham_dataset.csv containing labeled email samples. The dataset is balanced to ensure equal representation of spam and non-spam emails during training.

## Model Architecture

- Embedding Layer: 32-dimensional word embeddings
- LSTM Layer: 16 units for sequence processing
- Dense Layer: 32 units with ReLU activation
- Output Layer: Single unit with sigmoid activation for binary classification

## Results

- Test Accuracy: 96.33%
- Test Loss: 0.1537

The model demonstrates strong performance in distinguishing between spam and legitimate emails, with high precision and recall for both classes.

## Requirements

- Python 3.12+
- TensorFlow 2.16.2
- NLTK
- WordCloud
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## Usage

Open and run the Jupyter notebook in the `notebook/` directory to:
1. Load and preprocess the dataset
2. Train the LSTM model
3. Evaluate performance metrics
4. Save the trained model

The trained model is saved in `models/spam_detection_model.keras` and can be loaded for future predictions.

## Project Structure

```
Email-Spam-Detection-ML/
├── data/
│   └── spam_ham_dataset.csv
├── models/
│   └── spam_detection_model.keras
├── notebook/
│   └── notebook.ipynb
└── README.md
```

