import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import gensim
import gensim.corpora as corpora

data = pd.read_csv('news_articles.csv')

print(data.head())


articles = data['content']


articles = articles.str.lower().apply(lambda x: re.sub(r"([^\w\s])", "", x))


en_stopwords = stopwords.words("english")

articles = articles.apply(lambda x: word_tokenize(x))

ps = PorterStemmer()

articles = articles.apply(lambda x: [ps.stem(token) for token in x])


print(articles)


dictionary = corpora.Dictionary(articles)

print(dictionary)


doc_tem = [dictionary.doc2bow(text) for text in articles]


print(doc_tem)


num_topics = 2

lda_model = gensim.models.LdaModel(
    corpus=doc_tem, id2word=dictionary, num_topics=num_topics)


print(lda_model.print_topics(num_topics=num_topics, num_words=5))
