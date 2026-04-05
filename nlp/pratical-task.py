import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re
import pandas as pd

nltk.download('wordnet')


data = pd.read_csv('tripadvisor_hotel_reviews.csv')


data.info()
print(data.head())


print(data["Review"][0])


data['review_lowercase'] = data['Review'].str.lower()


print(data.head())


en_stopwords = stopwords.words("english")

en_stopwords.remove("not")


data['review_no_stopwords'] = data['review_lowercase'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in (en_stopwords)]))

print(data['review_no_stopwords'][0])

data['review_no_stopwords_no_punct'] = data.apply(
    lambda x: re.sub(r"[*]", "star", x['review_no_stopwords']), axis=1)
print(data.head())


data['tokenized'] = data.apply(lambda x: word_tokenize(
    x['review_no_stopwords_no_punct']), axis=1)
print(data["tokenized"][0])


ps = PorterStemmer()

data["stemmed"] = data['tokenized'].apply(
    lambda tokens: [ps.stem(token) for token in tokens])


print(data.head())


lemmatizer = WordNetLemmatizer()


data['lemmatized'] = data['tokenized'].apply(
    lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])


print(data['lemmatized'][0])

tokens_clean = sum(data['lemmatized'], [])

unigrams = (pd.Series(nltk.ngrams(tokens_clean, 1)).value_counts())


print(unigrams)
