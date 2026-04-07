if __name__ == "__main__":
    import pandas as pd
    import re
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
    from gensim.models import LsiModel
    from gensim.models.coherencemodel import CoherenceModel
    import gensim.corpora as corpora
    import matplotlib.pyplot as plt

    data = pd.read_csv('news_articles.csv')

    print(data.head())

    articles = data['content']

    articles = articles.str.lower().apply(
        lambda x: re.sub(r"([^\w\s])", "", x))

    en_stopwords = stopwords.words("english")

    articles = articles.apply(lambda x: word_tokenize(x))

    ps = PorterStemmer()

    articles = articles.apply(lambda x: [ps.stem(token) for token in x])

    print(articles)

    dictionary = corpora.Dictionary(articles)

    print(dictionary)

    doc_tem = [dictionary.doc2bow(text) for text in articles]

    num_topics = 2

    lsa_model = LsiModel(
        corpus=doc_tem, id2word=dictionary, num_topics=num_topics)

    print(lsa_model.print_topics(num_topics=num_topics, num_words=5))

    coherence_values = []
    model_list = []

    min_topics = 2
    max_topics = 11

    for num_topics in range(min_topics, max_topics+1):
        model = LsiModel(
            corpus=doc_tem,
            id2word=dictionary,
            num_topics=num_topics,
            random_seed=0)
        model_list.append(model)
        coherence_model = CoherenceModel(
            model=model, texts=articles, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherence_model.get_coherence())

    plt.plot(range(min_topics, max_topics+1), coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence Score")
    plt.title("Optimal Number of Topics")
    plt.show()
