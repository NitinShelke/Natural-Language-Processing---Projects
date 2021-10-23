import nltk
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

pattern = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

def preProcessTest(news):
    ps=PorterStemmer()
    news=re.sub(pattern,' ',news)
    news=news.lower()
    words=nltk.word_tokenize(news)
    words=[word for word in words if word not in set(stopwords.words('english'))]
    words=[ps.stem(word) for word in words]
    news=' '.join(words)
    
    return news

def preProcessTrain():
    data=pd.read_csv("data/train/kaggle_fake_train.csv")
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    
    data=data.drop(['id','author','text'],axis=1)
    data.reset_index(inplace=True)
    
    ps=PorterStemmer()
    
    corpus=[]
    for i in range(0,data.shape[0]):
        sent=re.sub(pattern,' ',data.title[i])
        sent=sent.lower()
        words=nltk.word_tokenize(sent)
        words=[word for word in words if word not in set(stopwords.words('english'))]
        words=[ps.stem(word) for word in words]
        sent=' '.join(words)
        corpus.append(sent)
    y=data['label']
    
    return corpus,y