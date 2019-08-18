#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 18:24:09 2019

@author: vikram
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 09:41:13 2019

@author: vikram

for  regular regular expression basics
https://www.geeksforgeeks.org/regular-expression-python-examples-set-1/
"""
import pandas as pd
import re
import nltk
#nltk.download('stopwords')
nltk.download('punkt')
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


from nltk.tokenize import sent_tokenize, word_tokenize 

features=pd.DataFrame()

def char_count(essay):
    p=re.compile('\w')
    char_count=p.findall(essay)
    return len(char_count)

def word_count(essay):
    p=re.compile('\w+')
    char_count=p.findall(essay)
    return len(char_count)

def sent_count(essay):
    s=sent_tokenize(essay) 
    return len(s)

def avg_word_le(essay):
    words =re.findall('\w+',essay)
    return sum(len(word) for word in words)/len(words)

def extract_features(essay):
    features['char_count']=essay['essay'].apply(char_count)
    features['word_count']=essay['essay'].apply(word_count)
    features['sent_count']=essay['essay'].apply(sent_count)
    features['avg_word_le']=essay['essay'].apply(avg_word_le)
    features['domain1_score']=essay['domain1_score']
    return features

# extracting features from essay set 1

features_set = extract_features(essay[essay['essay_set'] == 1])

print(features_set)





from sklearn.feature_extraction.text import CountVectorizer
def get_count_vectors(essays):
    
    vectorizer = CountVectorizer(max_features = 10000, ngram_range=(1, 3), stop_words='english')
    
    count_vectors = vectorizer.fit_transform(essays)
    
    feature_names = vectorizer.get_feature_names()
    
    return feature_names, count_vectors
    
feature_names_cv, count_vectors=get_count_vectors(essay[essay['essay_set'] == 1]['essay'])       

X_cv = count_vectors.toarray()

##############features bow+extract features concatenate#########
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import cohen_kappa_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
X = np.concatenate((features_set.iloc[:, 0:4].as_matrix(), X_cv), axis = 1)
y = features_set['domain1_score'].as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


linear_regressor = LinearRegression()

linear_regressor.fit(X_train, y_train)

y_pred = linear_regressor.predict(X_test)

# The coefficients
print('Coefficients: \n', linear_regressor.coef_)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % linear_regressor.score(X_test, y_test))

# Cohen’s kappa score: 1 is complete agreement
print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred), y_test))





# Exploratory Data Analysis (EDA) on the data
#import matplotlib as plot
features_set.plot.scatter(x = 'char_count', y = 'domain1_score', s=10)
features_set.plot.scatter(x = 'word_count', y = 'domain1_score', s=10)
features_set.plot.scatter(x = 'sent_count', y = 'domain1_score', s=10)
features_set.plot.scatter(x = 'avg_word_le', y = 'domain1_score', s=10)
#features_set1.plot.scatter(x = 'lemma_count', y = 'domain1_score', s=10)
#features_set1.plot.scatter(x = 'spell_err_count', y = 'domain1_score', s=10)
#features_set1.plot.scatter(x = 'noun_count', y = 'domain1_score', s=10)
#features_set1.plot.scatter(x = 'adj_count', y = 'domain1_score', s=10)
#features_set1.plot.scatter(x = 'verb_count', y = 'domain1_score', s=10)
#features_set1.plot.scatter(x = 'adv_count', y = 'domain1_score', s=10)














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



# import the existing word and sentence tokenizing 
# libraries 
from nltk.tokenize import sent_tokenize, word_tokenize 

text = "Natural language processing (NLP) is a field"  "of computer science, artificial intelligence"
	
print(len(sent_tokenize(text)) )
#print(word_tokenize(text))` 






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