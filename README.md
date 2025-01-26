# sentiment_analysis_Amazon
## Problem Statement

Amazon aims to enhance its product offerings, improve customer satisfaction, and drive better overall business outcomes. To achieve this, it is crucial to extract valuable insights from customer reviews, uncovering patterns, trends, and sentiments hidden within the data.

## Objective

This project focuses on developing a sentiment analysis model capable of automatically categorizing customer reviews based on their sentiment, identifying product trends, and pinpointing common complaints or praises. The final deliverable is a detailed report that provides actionable insights, leveraging advanced machine learning and natural language processing techniques.

## Evaluation Metric

The model’s performance will be assessed using accuracy, defined as the ratio of correct predictions to the total number of predictions. This metric will help determine how effectively the model labels reviews.

## Installation

To set up the project environment, install the required dependencies:
```bash
pip install datasets transformers torch umap-learn bertopic wordcloud scikit-learn matplotlib seaborn accelerate huggingface_hub
```
Imports used in the project:
```bash
from datasets import Dataset, DatasetDict, ClassLabel, load_from_disk
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments, pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from wordcloud import WordCloud
from bertopic import BERTopic
import umap.umap_ as umap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
```


# Below are a few examples of how to use the project:
```bash
#Identify Columns for Removal:
print(columns_to_remove)

#Convert input_ids into their corresponding text:
tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print(tokens)
print(tokenizer.convert_tokens_to_string(tokens))

#Train/Test Split Example:
#Refer to the code in the notebook for splitting datasets and training models with appropriate arguments.
```
# Project Highlights

Utilized cutting-edge NLP frameworks, such as Hugging Face Transformers, to perform sentiment analysis with high accuracy. Applied data visualization methods like UMAP for dimensionality reduction and word clouds to uncover key patterns in customer reviews. Incorporated machine learning approaches, including logistic regression and transformer-based classifiers, to enhance predictive performance.  
