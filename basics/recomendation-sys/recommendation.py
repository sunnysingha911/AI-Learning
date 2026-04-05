import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import ast
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('data/tmdb_5000_movies.csv')
credits = pd.read_csv('data/tmdb_5000_credits.csv')


# print(movies.head(2))
# print(credits.head(2))


movies = movies.merge(credits, on="title")


movies = movies[['movie_id', 'title', 'overview',
                 'genres', 'keywords', 'cast', 'crew']]


movies.dropna(inplace=True)


print(movies.duplicated().sum())


def convert(text):
    l = []
    for i in ast.literal_eval(text):
        l.append(i['name'])
    return l


def convert_cast(text):
    l = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            l.append(i['name'])
        counter += 1
    return l


def fetch_director(text):
    l = []

    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            l.append(i['name'])
            break
    return l


def remove_space(word):
    l = []
    for i in word:
        l.append(i.replace(" ", ""))
    return l


movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert_cast)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())


movies['cast'] = movies['cast'].apply(remove_space)
movies['crew'] = movies['crew'].apply(remove_space)
movies['genres'] = movies['genres'].apply(remove_space)
movies['keywords'] = movies['keywords'].apply(remove_space)
movies['tags'] = movies['overview'] + movies['genres'] + \
    movies['keywords'] + movies['cast'] + movies['crew']


new_df = movies[['movie_id', 'title', 'tags']]

new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

ps = PorterStemmer()


def stems(text):
    l = []
    for i in text.split():
        l.append(ps.stem(i))

    return " ".join(l)


new_df['tags'] = new_df['tags'].apply(stems)

cv = CountVectorizer(max_features=5000, stop_words='english')

vector = cv.fit_transform(new_df['tags']).toarray()

print(vector)
print(vector.shape)

similarity = cosine_similarity(vector)

# print(similarity)


def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distance = sorted(
        enumerate(similarity[index]), reverse=True, key=lambda x: x[1])
    for i in distance[1:6]:
        print(new_df.iloc[i[0]].title)


recommend("The Dark Knight Rises")


pickle.dump(new_df, open('artifacts/movie_list.pkl', 'wb'))
pickle.dump(similarity, open('artifacts/similarity.pkl', 'wb'))

new_df.head(2).to_html("movies_table.html")
