# Sentiment Analysis with Machine Learning

This project focuses on performing sentiment analysis on Twitter data using natural language processing (NLP) and machine learning techniques. The objective is to build a model that can classify the sentiment of tweets into categories such as positive, negative, or neutral.

## Table of Contents
1. [Introduction](#introduction)
2. [Setup and Libraries](#setup-and-libraries)
3. [Load Data](#load-data)
4. [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
5. [Text Visualization](#text-visualization)
6. [Train-Test Split](#train-test-split)
7. [Initialize TF-IDF Vectorizer](#initialize-tfidf-vectorizer)
8. [Build and Evaluate Model](#build-and-evaluate-model)
9. [SHAP Analysis](#shap-analysis)
10. [Sentiment Prediction](#sentiment-prediction)
11. [Conclusion](#conclusion)
12. [Author](#author)
13. [License](#license)

## Introduction

Sentiment analysis, also known as opinion mining, involves analyzing text data to determine the sentiment expressed by the author. This project uses a dataset of tweets to train a logistic regression model that classifies sentiments into categories such as positive, negative, or neutral.

## Setup and Libraries

First, we need to set up the environment and import the necessary libraries. This includes libraries for data manipulation, visualization, natural language processing, machine learning, and SHAP for model interpretability.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import shap

# Download necessary NLTK data
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Initialize stop words and stemmer
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
```

## Load Data

We load the dataset from a CSV file, dropping unnecessary columns and handling any missing values.

```python
# Load Data
column_names = ['id', 'topic', 'sentiment', 'text']
file_path = "/mnt/data/twitter_training.csv"
df = pd.read_csv(file_path, names=column_names)

# Drop 'id' column
df.drop('id', axis=1, inplace=True)

# Remove rows with NaN values in the 'sentiment' column
df = df.dropna(subset=['sentiment'])

# Inspect the first few rows and columns
df.head()
print(df.columns)
```

## Data Cleaning and Preprocessing

Text data needs to be cleaned and preprocessed before it can be used for training a machine learning model. This involves removing punctuation, converting text to lowercase, tokenizing, removing stop words, and stemming.

```python
# Data Cleaning and Preprocessing

def preprocess_text(text):
    # Convert to string
    text = str(text)
    
    # Removing punctuation, numbers, and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize the text
    tokens = text.split()
    
    # Remove stop words and apply stemming
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Apply the preprocess_text function to the 'text' column
df['text'] = df['text'].apply(preprocess_text)
df.head()
```

## Text Visualization

Visualizing the text data helps in understanding its structure and distribution. We use word clouds and sentiment distribution plots for this purpose.

```python
# Text Visualization

# Generate Word Cloud
all_words = ' '.join([text for text in df['text']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# Plot the distribution of sentiments
plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment', data=df)
plt.title('Sentiment Distribution')
plt.show()
```

## Train-Test Split

We split the data into training and testing sets to evaluate the performance of our model.

```python
# Train-Test Split
X = df['text']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Initialize TF-IDF Vectorizer

The TF-IDF vectorizer is used to convert the text data into numerical features that can be fed into the machine learning model.

```python
# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
```

## Build and Evaluate Model

We train a logistic regression model and evaluate its performance using various metrics such as accuracy, precision, recall, and F1 score. We also visualize the confusion matrix.

```python
# Build and Evaluate Model
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()
```

## SHAP Analysis

SHAP (SHapley Additive exPlanations) is used for interpreting the predictions of our machine learning model. It helps in understanding the impact of each feature on the model's output.

```python
# SHAP Analysis
explainer = shap.LinearExplainer(model, X_train_tfidf)
shap_values = explainer.shap_values(X_test_tfidf)

shap.summary_plot(shap_values, X_test_tfidf, feature_names=tfidf_vectorizer.get_feature_names_out())
```

## Sentiment Prediction

We define a function to predict the sentiment of new text inputs using the trained model.

```python
# Sentiment Prediction

def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)

    # Transform the preprocessed text into TF-IDF vectors
    X_new = tfidf_vectorizer.transform([preprocessed_text])

    # Make predictions
    predicted_sentiment = model.predict(X_new)

    # Print the predicted sentiment
    print("Predicted sentiment:", predicted_sentiment)

predict_sentiment("nothing")
predict_sentiment("Nvidia is a Giant Company")
predict_sentiment("Die you son of something")
```

## Conclusion

This project demonstrates how to preprocess text data, visualize it, build a machine learning model for sentiment analysis, and evaluate its performance. Additionally, it shows how to interpret the model's predictions using SHAP values. Feel free to experiment with the provided code and adapt it to your needs.

## Author

[Omar Elborollosy]

## License

This project is licensed under the MIT License - see the LICENSE file for details.
