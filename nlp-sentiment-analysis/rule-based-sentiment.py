from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sentence_1 = "i had a great time at the movie it was really funny"
sentence_2 = "i had a great time at the movie but the parking was terrible"
sentence_3 = "i had a great time at the movie but the parking wasn't great"
sentence_4 = "i went to see a movie"


print(sentence_1)
sentiment_score_1 = TextBlob(sentence_1)
print(sentiment_score_1.polarity)


print(sentence_2)
sentiment_score_2 = TextBlob(sentence_2)
print(sentiment_score_2.polarity)


print(sentence_3)
sentiment_score_3 = TextBlob(sentence_3)
print(sentiment_score_3.polarity)


print(sentence_4)
sentiment_score_4 = TextBlob(sentence_4)
print(sentiment_score_4.polarity)


vader_sentiment = SentimentIntensityAnalyzer()

print(vader_sentiment.polarity_scores(sentence_1))
print(vader_sentiment.polarity_scores(sentence_2))
print(vader_sentiment.polarity_scores(sentence_3))
print(vader_sentiment.polarity_scores(sentence_4))
