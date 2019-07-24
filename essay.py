#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 09:41:13 2019

@author: vikram
"""
import pandas as pd
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
essay = pd.read_csv('training_set_rel3.tsv', sep='\t', header=0,encoding="latin-1")  
essay.head()
essay.info()

from nltk.stem.porter import PorterStemmer
corpus  = []

for i in range(0, 3):
    review = re.sub('[^a-zA-Z]', ' ', essay['essay'][i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    
    #lem = WordNetLemmatizer() #Another way of finding root word
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review]
    #review = [lem.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
# Also known as the vector space model
# Text to Features (Feature Engineering on text data)
    
from sklearn.feature_extraction.text import CountVectorizer

  #bOW
# list of text documents
text = ["The quick brown fox jumped over the lazy dog."]
# create the transform
vectorizer = CountVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())


#Word Frequencies with TfidfVectorizer
	
from sklearn.feature_extraction.text import TfidfVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog.",
		"The dog.",
		"The fox"]
# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
# encode document
vector = vectorizer.transform([text[0]])
# summarize encoded vector
print(vector.shape)
print(vector.toarray())