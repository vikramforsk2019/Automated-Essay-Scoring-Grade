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
#from nltk.corpus import stopwords

essay = pd.read_csv('training_set_rel3.tsv', sep='\t', header=0,encoding="latin-1")

essay=essay[['essay_id','essay_set','essay','domain1_score']]
essay=essay.dropna(axis=0)
essay.head()
essay.info()

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


#########vectorization by using TF-IDF###########
#Word Frequencies with TfidfVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
def get_count_vectors(essays):
    # create the transform
    print(essays[0])
    vectorizer = TfidfVectorizer(max_features = 10000,min_df=0,max_df=1000, ngram_range=(1, 3), stop_words='english')
    # tokenize and build vocab
    vectorizer.fit(essays)
    feature_names = vectorizer.get_feature_names()
    #tf_cv = count_vectors.toarray()
    # encode document
    vector = vectorizer.transform(essays)
    # summarize encoded vector
    print(vector.shape)
    #print(vector.toarray())
    #count_vectors = vector.toarray()
    return feature_names,vector

#####vectorization by bow###############
"""
from sklearn.feature_extraction.text import CountVectorizer
def get_count_vectors(essays):
    
    vectorizer = CountVectorizer(max_features = 10000, ngram_range=(1, 3), stop_words='english')
    
    count_vectors = vectorizer.fit_transform(essays)
    
    feature_names = vectorizer.get_feature_names()
    
    return feature_names, count_vectors
   """
feature_names_cv, count_vectors=get_count_vectors(essay[essay['essay_set'] == 1]['essay'])       

X_cv = count_vectors.toarray()

##############features bow+extract features concatenate#########
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression #, Ridge, Lasso
#from sklearn.svm import SVR
#from sklearn import ensemble
#from sklearn.model_selection import GridSearchCV
from sklearn.metrics import cohen_kappa_score

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




################svm############
from sklearn.svm import SVC
classifier = SVC(kernel = 'poly', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
labels_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, labels_pred)

# Model Score
score = classifier.score(X_test,y_test)

print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(labels_pred), y_test))

"""
from sklearn.model_selection import GridSearchCV
svr = SVC()

parameters = {'kernel':['linear', 'rbf'], 'C':[1, 100], 'gamma':[0.1, 0.001]}

grid = GridSearchCV(svr, parameters)
grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)

# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_estimator_)

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % grid.score(X_test, y_test))

# Cohen’s kappa score: 1 is complete agreement
print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred), y_test))


"""











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








#Word Frequencies with TfidfVectorizer
	
from sklearn.feature_extraction.text import TfidfVectorizer
#list of text documents
text = ["The quick brown fox jumped over the lazy dog.",
		"The dog.",
		"The fox"]
# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
count_vectors = vectorizer.fit(text)
feature_names = vectorizer.get_feature_names()
#tf_cv = count_vectors.toarray()
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print(vector.shape)
print(vector.toarray())















