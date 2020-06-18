# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 02:47:12 2020

@author: Anes ABBAD

@website : http://abbadanes.github.io
"""


import pandas as pd
import string
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import nltk
nltk.download('punkt')
from nltk import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def preprocess(tokens):
    stop_words = []
    for c in string.punctuation:
        stop_words.append(c)
    stop_words.extend(nltk.corpus.stopwords.words('french'))
    for i in range(len(tokens)):
        tokens[i] = [y for y in tokens[i] if y not in stop_words]
    return tokens

def apply_topics(tokens):
    dictionary = Dictionary(tokens)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in tokens]
    ldamodel = LdaModel(doc_term_matrix, num_topics=1 ,id2word = dictionary, passes=50,random_state=0)
    return ldamodel

def get_words_topic(model):
    for index, topic in model.show_topics(formatted=False, num_words= 10):
        mots = [w[0] for w in topic]
        freq = [int(w[1] *1000) for w in topic]
    mots = ' '.join(mots)
    return mots



dataset = pd.read_csv("dataset.csv")

for i in dataset.source.unique():
    df = dataset.loc[dataset.source == i]
    tokens = [word_tokenize(str(i)) for i in df["content"].tolist()]
    tokens = preprocess(tokens)    
    model = apply_topics(tokens)    
    mots = get_words_topic(model)
    wordcloud = WordCloud(max_font_size=50, max_words=10, background_color="black").generate(mots)
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("Topic model on - "+str(i))
    plt.savefig("Topic model on - "+str(i)+".png")
    plt.show()