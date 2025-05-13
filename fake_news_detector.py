# -*- coding: utf-8 -*-

!pip install wordcloud seaborn scikit-learn streamlit -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

!pip install nltk -q

import nltk
from nltk.corpus import stopwords
from collections import Counter

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO

from google.colab import files
uploaded = files.upload()

true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

true_df['label'] = 1
fake_df['label'] = 0

df = pd.concat([true_df, fake_df], ignore_index=True)

df = df.sample(frac=1).reset_index(drop=True)

df.head()

df = df[['title', 'text', 'label']]

df = df.fillna("")

df['content'] = df['title'] + " " + df['text']

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

df['content'] = df['content'].apply(clean_text)
df.head()

fake_words = ' '.join(df[df.label == 0]['content'])
true_words = ' '.join(df[df.label == 1]['content'])

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.imshow(WordCloud().generate(fake_words))
plt.title("Fake News WordCloud")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(WordCloud().generate(true_words))
plt.title("Real News WordCloud")
plt.axis("off")
plt.show()

# Train-test split
X = df['content']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_tfidf, y_train)
y_pred_lr = lr.predict(X_test_tfidf)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train_tfidf, y_train)
y_pred_rf = rf.predict(X_test_tfidf)

# Evaluation
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.show()

from PIL import Image

def generate_wordcloud(text):
    wc = WordCloud(width=600, height=300, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)  # prevent memory leaks
    buf.seek(0)
    return Image.open(buf)  # ✅ Return as PIL Image

def generate_bar_chart(text):
    words = [word.lower() for word in text.split() if word.isalpha() and word.lower() not in stop_words]
    counter = Counter(words)
    most_common = counter.most_common(10)

    if not most_common:
        return None

    labels, values = zip(*most_common)
    fig, ax = plt.subplots()
    ax.barh(labels, values, color='skyblue')
    ax.set_title("Top 10 Word Frequencies")
    ax.invert_yaxis()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def predict_with_visuals(text):
    if not text.strip():
        return "❌ Please enter some text.", None, None

    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    label = "✅ Real News" if prediction == 1 else "❌ Fake News"

    wc_image = generate_wordcloud(text)
    bar_image = generate_bar_chart(text)
    return label, wc_image, bar_image

!pip install gradio -q
import gradio as gr

gr.Interface(
    fn=predict_with_visuals,
    inputs=gr.Textbox(lines=5, placeholder="Paste news content here..."),
    outputs=[
        gr.Text(label="Prediction"),
        gr.Image(label="WordCloud"),
        gr.Image(label="Top Word Frequencies")
    ],
    title="📰 Fake News Detector with Visual Analysis",
    description="Enter news content to detect fake news. Get a prediction, wordcloud, and top word frequency bar chart."
).launch(debug=True)