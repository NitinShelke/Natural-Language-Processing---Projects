from sklearn.feature_extraction.text import CountVectorizer
import pickle
from com_in_nitin_utils.utils import preProcessTest

def predict(news):
    news=preProcessTest(news)
    cv=pickle.load(open("model/vectorizer.pkl","rb"))
    news=cv.transform([news]).toarray()
    
    model=pickle.load(open("model/model.pkl","rb"))
    
    prediction=model.predict(news)
    
    return prediction
    