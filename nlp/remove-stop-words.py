
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")


en_stopwords = stopwords.words("english")


sentence = "it was too far to go to the shop and he did not want her to walk"

sentence_no_stop_words = " ".join(
    word for word in sentence.split() if word not in en_stopwords)

# print(en_stopwords)


print(sentence_no_stop_words)
