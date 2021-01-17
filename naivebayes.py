from sklearn.naive_bayes import MultinomialNB
import preprocess as prep
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

#%% Data1 "Text"

data1 = prep.data_text
y = data1["status"]
X = data1["text"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=42)

vectorize = TfidfVectorizer()
vectorize.fit(X_train)
Xv_train = vectorize.transform(X_train)
Xv_test=vectorize.transform(X_test)

#%% Data2 "Title"
data2 = prep.data_title

y2 = data2["status"]
X2 = data2["title"]
X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.3, random_state=42) #splitting the same

vectorize2 = TfidfVectorizer()
vectorize2.fit(X2_train)
Xv2_train = vectorize2.transform(X2_train)
Xv2_test=vectorize2.transform(X2_test)

#%% Naive Bayes
nb=MultinomialNB()
def NB(X_train, y_train, X_test, y_test): #send vectorized
    nb.fit(X_train, y_train)
    pred = nb.predict(X_test)
    score = nb.score(X_test, y_test)
    return pred, round(score,3)

#%% Plot for heatmap of confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
def heatmap_cm(clf,X, y, acc, name):
    plot_confusion_matrix(clf, X, y)
    plt.title('{0} Accuracy Score: {1}'.format(name,acc), size = 15)
    print("check")
    plt.show()

#%%%%

text_pred, text_score = NB(Xv_train, y_train, Xv_test, y_test)
heatmap_cm(nb, Xv_test, y_test, text_score, "Naive Bayes Text")
print("Title")
title_pred, title_score = NB(Xv2_train, y2_train, Xv2_test, y2_test)
print(title_score)
print(Xv2_train.shape)
print(y2_train.shape)
heatmap_cm(nb, Xv2_test, y2_test, title_score, "Naive Bayes Title")

