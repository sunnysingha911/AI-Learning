
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
nltk.download("punkt_tab")


sentence = "Her cat's name is Luna. Her dog's name is max"

print(sent_tokenize(sentence))


sentence = "her cat's name is luna"

print(word_tokenize(sentence))


sentence2 = "Her cat's name is Luna and her dog's name is max"

print(word_tokenize(sentence2))
