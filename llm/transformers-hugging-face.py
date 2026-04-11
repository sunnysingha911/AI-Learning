from transformers import pipeline;

sentiment_classifier = pipeline('sentiment-analysis')

sentiment = sentiment_classifier("I'm so excited to be learning about large language models")

print(sentiment)


ner = pipeline('ner', model="dslim/bert-base-NER")


print(ner('Her name is Anna and she works in New York City for Morgan Stanley'))


zeroshot_classifier  = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

sequence_to_classify  = "one day I will see the world"

candidate_labels = ['travel','cooking', 'dancing']

print(zeroshot_classifier(sequence_to_classify, candidate_labels))