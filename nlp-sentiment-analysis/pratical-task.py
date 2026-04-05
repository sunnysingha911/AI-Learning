import pandas as pd
import numpy as np
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import transformers
from transformers import pipeline
import matplotlib.pyplot as plt


data = pd.read_csv('book_reviews_sample.csv')


# print(data.head())

# print(data['reviewText'][0])


data['reviewText_clean'] = data.apply(lambda x: re.sub(
    r"([^\w\s])", "", x["reviewText"].lower()), axis=1)


vader_sentiment = SentimentIntensityAnalyzer()

data['vader_sentiment_score'] = data['reviewText_clean'].apply(
    lambda x: vader_sentiment.polarity_scores(x)['compound'])

print(data.head())


bins = [-1, -0.1, 0.1, 1]

names = ['negative', 'neutral', 'positive']


data['vader_sentiment_label'] = pd.cut(
    data['vader_sentiment_score'], bins, labels=names)


# data['vader_sentiment_label'].value_counts().plot.bar()
# plt.show()


transformers_pipeline = pipeline("sentiment-analysis")

transformers_labels = []


for review in data['reviewText_clean'].values:
    sentiment_list = transformers_pipeline(review)
    sentiment_label = [sent['label'] for sent in sentiment_list]
    transformers_labels.append(sentiment_label)

data['transformers_sentiment_labels'] = transformers_labels
data['transformers_sentiment_labels'].value_counts().plot.bar()
plt.show()
