if __name__ == "__main__":

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import spacy
    from spacy import displacy
    from spacy import tokenizer
    import re
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.corpus import stopwords
    import spacy.tokens
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import gensim
    import gensim.corpora as corpora
    from gensim.models.coherencemodel import CoherenceModel
    from gensim.models import LsiModel, TfidfModel
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.metrics import accuracy_score, classification_report

    plt.rcParams['figure.figsize'] = (12, 8)
    default_plot_colour = '#00bfbf'

    data = pd.read_csv('fake_news_data.csv')

    # print(data.head())
    # print(data.info())
    # data['fake_or_factual'].value_counts().plot(
    #     kind='bar', color=default_plot_colour)
    # plt.title('Count of Article Classification')

    # POS tagging
    nlp = spacy.load('en_core_web_sm')

    fake_news = data[data['fake_or_factual'] == 'Fake News']
    fact_news = data[data['fake_or_factual'] == 'Factual News']

    fake_spacydocs = list(nlp.pipe(fake_news['text']))
    fact_spacydocs = list(nlp.pipe(fact_news['text']))

    def extract_token_tags(doc: spacy.tokens.doc.Doc):
        return [([i.text, i.ent_type_, i.pos_]) for i in doc]

    fake_tagsdf = []
    columns = ["token", "ner_tag", "pos_tag"]

    for ix, doc in enumerate(fake_spacydocs):
        tags = extract_token_tags(doc)
        tags = pd.DataFrame(tags)
        tags.columns = columns
        fake_tagsdf.append(tags)

    fake_tagsdf = pd.concat(fake_tagsdf)

    fact_tagsdf = []

    for ix, doc in enumerate(fact_spacydocs):
        tags = extract_token_tags(doc)
        tags = pd.DataFrame(tags)
        tags.columns = columns
        fact_tagsdf.append(tags)

    fact_tagsdf = pd.concat(fact_tagsdf)

    pos_count_fake = fake_tagsdf.groupby(['token', 'pos_tag']).size().reset_index(
        name="counts").sort_values(by="counts", ascending=False)

    pos_count_fact = fact_tagsdf.groupby(['token', 'pos_tag']).size().reset_index(
        name="counts").sort_values(by="counts", ascending=False)

    pos_count_fake_gb_tag = pos_count_fake.groupby("pos_tag")[
        'token'].count().sort_values(ascending=False).head(10)

    pos_count_fact_gb_tag = pos_count_fact.groupby("pos_tag")[
        'token'].count().sort_values(ascending=False).head(10)

    pos_count_fake_noun = pos_count_fake[pos_count_fake.pos_tag == "NOUN"][:15]
    pos_count_fact_noun = pos_count_fact[pos_count_fact.pos_tag == "NOUN"][:15]

    # named entities
    top_entities_fake = fake_tagsdf[fake_tagsdf['ner_tag'] != ""].groupby(
        ['token', 'ner_tag']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)

    top_entities_fact = fact_tagsdf[fact_tagsdf['ner_tag'] != ""].groupby(
        ['token', 'ner_tag']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)

    ner_palette = {
        'ORG': sns.color_palette("Set2").as_hex()[0],
        'GPE': sns.color_palette("Set2").as_hex()[1],
        'NORP': sns.color_palette("Set2").as_hex()[2],
        'PERSON': sns.color_palette("Set2").as_hex()[3],
        'DATE': sns.color_palette("Set2").as_hex()[4],
        'CARDINAL': sns.color_palette("Set2").as_hex()[5],
        'PERCENT': sns.color_palette("Set2").as_hex()[6],
    }

    # sns.barplot(
    #     x='counts',
    #     y='token',
    #     hue='ner_tag',
    #     palette=ner_palette,
    #     data=top_entities_fake[:10],
    #     orient='h', dodge=False
    # ).set(title="Most Common Named Entities in Fake News")

    data['text_clean'] = data.apply(
        lambda x: re.sub(r"^[^-]*-\s", "", x['text']), axis=1)

    data['text_clean'] = data['text_clean'].str.lower()

    data['text_clean'] = data.apply(
        lambda x: re.sub(r"([^\w\s])", "", x['text_clean']), axis=1)

    en_stopwords = stopwords.words('english')

    data['text_clean'] = data['text_clean'].apply(lambda x: " ".join(
        [word for word in x.split() if word not in en_stopwords]))

    data['text_clean'] = data.apply(
        lambda x: word_tokenize(x['text_clean']), axis=1)

    lemmatizer = WordNetLemmatizer()

    data['text_clean'] = data['text_clean'].apply(
        lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])

    # flattens a array inside array [[1, 2], [3, 4]] -> [1, 2, 3, 4]
    tokens_clean = sum(data['text_clean'], [])

    # print(tokens_clean)

    unigrams = (pd.Series(nltk.ngrams(tokens_clean, 1)
                          ).value_counts()).reset_index()[:10]

    unigrams['token'] = unigrams['index'].apply(lambda x: x[0])

    # sns.barplot(x='count', y='token', data=unigrams, orient='h',
    #             palette=[default_plot_colour], hue='token', legend=False).set(title="Most Common Unigrams after pre processing")

    print(unigrams)

    # sentiment analysis

    vader_sentiment = SentimentIntensityAnalyzer()

    data['vader_sentiment_score'] = data['text'].apply(
        lambda x: vader_sentiment.polarity_scores(x)['compound'])

    bins = [-1, -0.1, 0.1, 1]

    names = ['negative', 'neutral', 'positive']

    data['vader_sentiment_label'] = pd.cut(
        data['vader_sentiment_score'], bins, labels=names)

    # data['vader_sentiment_label'].value_counts().plot.bar(
    #     color=default_plot_colour)

    # topic modelling
    fake_news_text = data[data['fake_or_factual'] ==
                          'Fake News']['text_clean'].reset_index(drop=True)

    dictionary_fake = corpora.Dictionary(fake_news_text)

    doc_term_fake = [dictionary_fake.doc2bow(text) for text in fake_news_text]

    coherence_values = []
    model_list = []

    min_topics = 2
    max_topics = 11

    for num_topics_i in range(min_topics, max_topics+1):
        model = gensim.models.LdaModel(
            doc_term_fake, num_topics=num_topics_i, id2word=dictionary_fake)
        model_list.append(model)
        coherence_model = CoherenceModel(
            model=model, texts=fake_news_text, dictionary=dictionary_fake, coherence='c_v')
        coherence_values.append(coherence_model.get_coherence())

    print(data.head())

    plt.plot(range(min_topics, max_topics+1), coherence_values)
    plt.xlabel("Number of topics")
    plt.ylabel("Coherence score")
    plt.legend(["coherence_values"], loc='best')

    num_topics_lda = 7
    lda_model = gensim.models.LdaModel(
        corpus=doc_term_fake, id2word=dictionary_fake, num_topics=num_topics_lda)

    print(lda_model.print_topics(num_topics=num_topics_lda, num_words=10))

    def tfidf_corpus(doc_term_matrix):
        tfidf = TfidfModel(corpus=doc_term_matrix, normalize=True)
        corpus_tfidf = tfidf[doc_term_matrix]
        return corpus_tfidf

    def get_coherence_score(corpus, dictionary, text, min_topics, max_topics):
        coherence_values = []
        model_list = []
        for num_topics_i in range(min_topics, max_topics+1):
            model = LsiModel(
                corpus, num_topics=num_topics_i, id2word=dictionary)
            model_list.append(model)
            coherence_model = CoherenceModel(
                model=model, texts=text, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherence_model.get_coherence())
        # plt.plot(range(min_topics, max_topics+1), coherence_values)
        # plt.xlabel("Number of topics")
        # plt.ylabel("Coherence score")
        # plt.legend(["coherence_values"], loc='best')
        # plt.show()

    corpus_tfidf_fake = tfidf_corpus(doc_term_fake)

    get_coherence_score(corpus_tfidf_fake, dictionary_fake,
                        fake_news_text, min_topics=2, max_topics=11)

    lsa_model = LsiModel(
        corpus_tfidf_fake, id2word=dictionary_fake, num_topics=7)

    print(lsa_model.print_topics())

    # custom classifier
    print(data.head())

    X = [','.join(map(str, l)) for l in data['text_clean']]

    Y = data['fake_or_factual']

    countvec = CountVectorizer()
    countvec_fit = countvec.fit_transform(X)
    bag_of_words = pd.DataFrame(
        countvec_fit.toarray(), columns=countvec.get_feature_names_out())

    X_train, X_test, Y_train, Y_test = train_test_split(
        bag_of_words, Y, test_size=0.3)

    lr = LogisticRegression(random_state=0).fit(X_train, Y_train)

    y_pred_lr = lr.predict(X_test)
    print(accuracy_score(y_pred_lr, Y_test))

    print(classification_report(Y_test, y_pred_lr))

    svm = SGDClassifier().fit(X_train, Y_train)

    y_pred_svm = svm.predict(X_test)

    print(accuracy_score(y_pred_svm, Y_test))

    print(classification_report(Y_test, y_pred_svm))

    print(X_train.head())
    print(Y_train.head())

    # plt.plot(range(2, 11+1), coherence_values)
    # plt.xlabel("Number of topics")
    # plt.show()
