# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 08:55:44 2021

@author: Koushik
"""

import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
import re


paragraph = """The news mentioned here is fake. Audience do not encourage fake news. Fake news is false or misleading"""

sentences = nltk.sent_tokenize(paragraph)

lemmatizer = WordNetLemmatizer()

corpus = []

# Lemmatization
for i in range(len(sentences)):
    sent = re.sub('[^a-zA-Z]', ' ', sentences[i])
    sent = sent.lower()
    sent= sent.split()
    sent = [lemmatizer.lemmatize(word) for word in sent if not word in set(stopwords.words('english'))]
    sent = ' '.join(sent)   
    corpus.append(sent)
 
    
print(corpus)


words_unique = []
for i in range(len(corpus)):
    unique = nltk.word_tokenize(corpus[i])
    words_unique.append(unique)

print(words_unique)

# Creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
independentFeatures_bow = cv.fit_transform(corpus).toarray()
print(independentFeatures_bow)


# Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
independentFeatures_tfIDF = tfidf.fit_transform(corpus).toarray()
print(independentFeatures_tfIDF)