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

essay=essay[['essay_id','essay_set','essay','domain1_score']]
essay=essay.dropna(axis=0)
essay.head()
essay.info()

from nltk.stem.porter import PorterStemmer
corpus  = []

for i in range(0, 1):
    review = re.sub('[^a-zA-Z]', ' ', essay['essay'][i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    #ps = PorterStemmer()
    #review = [ps.stem(word) for word in review]
    corpus.append(review)
    
my_dict={}
for i in corpus[0]:
    if i not in my_dict:
        my_dict[i]=1
    else:
        my_dict[i]+=1
print(len(my_dict))   

#tokens = nltk.word_tokenize(essay['essay'][0])
     
###########features extract##############




features=pd.DataFrame()

def char_count(essay):
    p=re.compile('\w')
    char_count=p.findall(essay)
    return len(char_count)

def word_count(essay):
    p=re.compile('\w+')
    char_count=p.findall(essay)
    return len(char_count)
    #tokens = nltk.word_tokenize(char_count)
    #print(tokens)
    
def extract_features(essay):
    features['char_count']=essay['essay'].apply(char_count)
    features['word_count']=essay['essay'].apply(word_count)
    
    return features

# extracting features from essay set 1

features_set = extract_features(essay[essay['essay_set'] == 1])

print(features_set)













import re 

# \d is equivalent to [0-9]. 
p = re.compile('\w+') 
print(len(p.findall("I went to him at 11 A.M. on 4th July 1886")) )



# \d+ will match a group on [0-9], group of one or greater size 
p = re.compile('\d+') 
print(p.findall("5I went to him at 11 A.M. on 4th July 1886")) 



p = re.sub('\d') 
print(p.findall("5I went to him at 11 A.M. on 4th July 1886")) 


re.sub('\s','','siuta3 df 3#22 ds2')


clean_essay = re.sub(r'\W', ' ', essay)









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