import re

my_folder = r"C:\desktop\notes"

print(my_folder)


result_search = re.search("pattern", r"string to contain the pattern")

print(result_search)


string = r"sara was able to help me find the items I needed quickly"

new_string = re.sub("sara", "sarah", string)


print(new_string)

customer_reviews = ["sam was a great help to me in the store",
                    "the cashier was very rude to me, I think her name was eleanor",
                    "amazing work from sadeen!",
                    "sarah was able to help me find the items I needed quickly",
                    "lucy is such a great addition to the team",
                    " great service from sara she found me what i wanted"]


sarah_reviews = []


pattern_to_find = r"sarah?"

for string in customer_reviews:
    if re.search(pattern_to_find, string):
        sarah_reviews.append(string)

print(sarah_reviews)


# starts with a

a_reviews = []


pattern_to_find = r"^a"

for string in customer_reviews:
    if (re.search(pattern_to_find, string)):
        a_reviews.append(string)

print(a_reviews)


# ends with y

y_reviews = []


pattern_to_find = r"y$"

for string in customer_reviews:
    if (re.search(pattern_to_find, string)):
        y_reviews.append(string)

print(y_reviews)


# needed or wanted in between sentence

need_want_reviews = []


pattern_to_find = r"(need|want)ed"

for string in customer_reviews:
    if (re.search(pattern_to_find, string)):
        need_want_reviews.append(string)

print(need_want_reviews)


# no punctuation

no_punct_reviews = []

pattern_to_find = r"[^\w\s]"

for string in customer_reviews:
    no_punct_string = re.sub(pattern_to_find, "", string)
    no_punct_reviews.append(no_punct_string)

print(no_punct_reviews)
