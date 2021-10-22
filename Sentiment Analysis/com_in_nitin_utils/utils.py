import json
import os
import shutil
import spacy
import re
import string
import pandas as pd

def getStopwords():
    stopwords=[]
    with open("data/stopwords.txt") as f:
        lines=f.read().splitlines()
    for line in lines:
        stopwords.append(line)
    return stopwords


def preprocessTrain(trainPath):
    stopwords=getStopwords()
   
    nlp = spacy.load("en_core_web_sm")
    pattern = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+\ "
    df = pd.DataFrame(columns=['target', 'text'])
    
    data=pd.read_json(trainPath)
    
    file=data['data']
    lable=[]
    text=[]
    for line in file:
        lable.append(line['lName'])
        text.append(line['lData'])
    
    for line in text:
        doc=nlp(line)
        clean_text=[]
        for i in doc:
            clean=str(re.sub(pattern,'',str(i.lemma_).lower().strip()))
            if clean not in stopwords:
                if clean not in string.punctuation:
                    clean_text.append(clean)
        df = df.append({'text': clean_text}, ignore_index=True)
    df['text'] = [" ".join(value) for value in df['text'].values]
    df['target']=lable
    
    return df


def preprocessTest(testPath):
    stopwords=getStopwords()
   
    nlp = spacy.load("en_core_web_sm")
    pattern = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+\ "
    df = pd.DataFrame(columns=['text'])
    
    data=pd.read_json(testPath)
    
    file=data['data']
    lable=[]
    text=[]
    for line in file:
        lable.append(line['lName'])
        text.append(line['lData'])
    
    for line in text:
        doc=nlp(line)
        clean_text=[]
        for i in doc:
            clean=str(re.sub(pattern,'',str(i.lemma_).lower().strip()))
            if clean not in stopwords:
                if clean not in string.punctuation:
                    clean_text.append(clean)
        df = df.append({'text': clean_text}, ignore_index=True)
    df['text'] = [" ".join(value) for value in df['text'].values]
    df['target']=lable
    
    return df