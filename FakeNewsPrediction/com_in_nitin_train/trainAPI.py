from com_in_nitin_utils.utils import preProcessTrain
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd

def train():
    X,y=preProcessTrain()
    cv=CountVectorizer()
   
    
    X=cv.fit_transform(X).toarray()
    pickle.dump(cv,open("model/vectorizer.pkl","wb"))
    lr=LogisticRegression()
    lr.fit(X,y)
    
    pickle.dump(lr,open("model/model.pkl","wb"))
    
    return("model trained and saved successfuly")
    