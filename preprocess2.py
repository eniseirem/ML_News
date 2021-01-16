import pandas as pd
from sklearn.utils import shuffle
import nltk
import string
import matplotlib.pyplot as plt


fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake['target'] = 'fake'
true['target'] = 'true'

data = pd.concat([fake, true]).reset_index(drop = True)

data = shuffle(data)
data = data.reset_index(drop=True)
data.drop(["date"], axis=1, inplace=True)
data.drop(["title"], axis=1, inplace=True)
data['text'] = data['text'].apply(lambda x: x.lower())

def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str
data['text'] = data['text'].apply(punctuation_removal)

nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

print(data.groupby(['subject'])['text'].count())
data.groupby(['subject'])['text'].count().plot(kind="bar")
plt.show()

print(data.groupby(['target'])['text'].count())
data.groupby(['target'])['text'].count().plot(kind="bar")
plt.show()