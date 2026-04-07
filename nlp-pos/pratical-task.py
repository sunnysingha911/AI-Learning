import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import re
import pandas as pd
import matplotlib.pyplot as plt


bbc_data = pd.read_csv("bbc_news.csv")
print(bbc_data.head())


titles = pd.DataFrame(bbc_data['title'])
# print(titles.head())


# lowercase
titles['lowercase'] = titles['title'].str.lower()


# stop word removal
en_stopwords = stopwords.words('english')
titles['no_stopwords'] = titles['lowercase'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in (en_stopwords)]))

# print(titles.head())


# punctuation removal
titles["no_stopwords_no_punct"] = titles.apply(
    lambda x: re.sub(r"[^\w\s]", "", x['no_stopwords']), axis=1)

# print(titles.head())

# tokenize
titles['token_raw'] = titles.apply(lambda x: word_tokenize(x['title']), axis=1)
titles['token_clean'] = titles.apply(
    lambda x: word_tokenize(x['no_stopwords_no_punct']), axis=1)

# print(titles.head())


# lemmatizing
lemmatizer = WordNetLemmatizer()

titles["tokens_clean_lemmatized"] = titles["token_clean"].apply(
    lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])

# print(titles.head())

# create list for just our tokens
tokens_raw_list = sum(titles["token_raw"], [])
tokens_clean_list = sum(titles["tokens_clean_lemmatized"], [])


print(titles.head())


# pos tagging
nlp = spacy.load("en_core_web_sm")

spacy_doc = nlp(" ".join(tokens_raw_list))

pos_df = pd.DataFrame(columns=["token", "pos_tag"])

for token in spacy_doc:
    pos_df = pd.concat([pos_df, pd.DataFrame.from_records(
        [
            {
                'token': token.text,
                'pos_tag': token.pos_
            }
        ]
    )], ignore_index=True)


pos_df_counts = pos_df.groupby(['token', 'pos_tag']).size().reset_index(
    name='count').sort_values(by='count', ascending=False)

print(pos_df_counts.head(10))


nouns = pos_df_counts[pos_df_counts.pos_tag == "NOUN"][:10]

print(nouns)


ner_df = pd.DataFrame(columns=['token', 'ner_tag'])


for token in spacy_doc.ents:
    if pd.isna(token.label) is False:
        ner_df = pd.concat([ner_df, pd.DataFrame.from_records([
            {'token': token.text,
             'ner_tag': token.label_}
        ])], ignore_index=True)
print(ner_df.head())


ner_df_count = ner_df.groupby(['token', 'ner_tag']).size().reset_index(
    name='count').sort_values(by='count', ascending=False)

print(ner_df_count.head(10))


people = ner_df_count[ner_df_count.ner_tag == 'PERSON'][:10]

print(people.head(10))
