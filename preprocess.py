import pandas as pd


df_f=pd.read_csv("data/Fake.csv")
df_f['status']=0 #we will keep the fake/true here.

df_t =pd.read_csv("data/True.csv")
df_t['status']=1

#merging datasets
data=pd.concat([df_f,df_t],axis=0)


# we will looking into to two datasets.
# One model predicting with text
#The other one predicting with only header of the news.

data_text=data.drop(['title','date'],axis=1)
data_title=data.drop(['text','date'],axis=1)

data.isnull().sum() #there is no empty variable


#%% Cleaning the text

import nltk #natural language toolkit
# >>> import nltk
# >>> nltk.download()
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
import re
import string
def del_punc (text):
    freepunc = "".join([i for i in text if i not in string.punctuation])
    return freepunc
def del_stops(text,s):
    stop = ' '.join([word for word in text if word not in s])
    return stop
lemmatizer = WordNetLemmatizer()
def lemma(text):
    lemm = [lemmatizer.lemmatize(word) for word in text]
    return lemm
tokenizer = RegexpTokenizer(r'\w+')

def clean_df(df): #send as data["text"]
    df = df.apply(lambda x: del_punc(x))
    print("Punctuation Removed")
    df = df.apply(lambda x: tokenizer.tokenize(x.lower()))
    print("Lowered")
    s = stopwords.words("english")
    df = df.apply(lambda x: del_stops(x,s))
    print("Stop words removed")
    lemm = [lemmatizer.lemmatize(word) for word in df]
    print("Lemmatized")
    return lemm


stops = stopwords.words("english")
data_text["text"] = clean_df(data_text["text"])
data_title["title"] = clean_df(data_title["title"])


