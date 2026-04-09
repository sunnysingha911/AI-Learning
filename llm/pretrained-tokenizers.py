from transformers import AutoTokenizer

model = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model)

sentence = "I'm so excited to be learning about large language models"

input_ids = tokenizer(sentence)

print(input_ids)

tokens = tokenizer.tokenize(sentence)

print(tokens)


token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(token_ids)


decoded_ids  = tokenizer.decode(token_ids)

print(decoded_ids)


model2 = "xlnet-base-cased"

tokenizer2 = AutoTokenizer.from_pretrained(model2)

input_ids = tokenizer2(sentence)

print(input_ids)


tokens = tokenizer2.tokenize(sentence)

print(tokens)

token_ids = tokenizer2.convert_tokens_to_ids(tokens)

print(token_ids)


from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch 


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

input_ids_pt = tokenizer(sentence, return_tensors = "pt")

print(input_ids_pt)

model  = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

with torch.no_grad():
    logits = model(**input_ids_pt).logits

predicted_class_id = logits.argmax().item()

print(model.config.id2label[predicted_class_id])


model_directory = "my_saved_models"

tokenizer.save_pretrained(model_directory)


model.save_pretrained(model_directory)


my_tokenizer = AutoTokenizer.from_pretrained(model_directory)


my_model = AutoModelForSequenceClassification.from_pretrained(model_directory)


print(my_model)