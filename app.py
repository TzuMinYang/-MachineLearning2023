import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_df.head()
train_df.info()

train_df.isnull().sum()
test_df.isnull().sum()

train_df.duplicated(["text", "target"]).sum()

train = train_df.drop(columns=['location'])
test = test_df.drop(columns=['location'])
train = train_df.drop(columns=['keyword'])
test = test_df.drop(columns=['keyword'])
train.head()
print(train.shape)

vectorizer = CountVectorizer(stop_words='english',ngram_range=(1, 3), min_df=1)
vectors = vectorizer.fit_transform(train['text'])
X = vectors.toarray()
Y = train['target'].values
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4285, random_state=67)
clf = LogisticRegression()
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
clf.score(X_test,y_test)
from sklearn.metrics import confusion_matrix,accuracy_score, recall_score, precision_score
accuracy_score(y_test,pred)
recall_score(y_test,pred)
precision_score(y_test,pred)

forSubmissionDF=pd.DataFrame(columns=['id','target'])
forSubmissionDF
print(pred.shape)
forSubmissionDF['id'] = test.id
forSubmissionDF['target'] = pred
forSubmissionDF.to_csv('for_submission_20230411.csv', index=False)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=150,bootstrap=True)
model2=rf.fit(X_train,y_train)
model2.score(X_train,y_train)
model2.score(X_test,y_test)
y_pred=model2.predict(X_test)
accuracy_score(y_test,y_pred)
recall_score(y_test,y_pred)
precision_score(y_test,y_pred)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(max_depth=98)
model1=dt.fit(X, Y)
model1.score(X_train,y_train)
model1.score(X_test,y_test)
d_pred=model1.predict(X_test)
accuracy_score(y_test,d_pred)
recall_score(y_test,d_pred)
precision_score(y_test,d_pred)

forSubmissionDF=pd.DataFrame(columns=['id','target'])
forSubmissionDF
print(pred.shape)
forSubmissionDF['id'] = test.id
forSubmissionDF['target'] = d_pred
forSubmissionDF.to_csv('for_submission_20230411_d.csv', index=False)