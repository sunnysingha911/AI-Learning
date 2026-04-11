from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import torch 

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

model = BertForQuestionAnswering.from_pretrained(model_name)

tokenizer = BertTokenizer.from_pretrained(model_name)

# EMBEDDINGS

question = "When was the first DVD released ?"

answer_document = "The first DVD(Digital Versatile Disc) was released on March 24, 1997. It was a movie titled 'Twister' and was released in Japan. "

encodings = tokenizer.encode_plus(text=question, text_pair=answer_document)

print(encodings)