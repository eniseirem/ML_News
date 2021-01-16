
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import preprocess as prep
from sklearn.feature_extraction.text import TfidfVectorizer

#%% Data1 "Text"

data1 = prep.data_text
y = data1["status"]
X = data1.drop(columns=["status","subject"])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=42)

vectorize = TfidfVectorizer()
Xv_train = vectorize.fit_transform(X_train)
Xv_test=vectorize.transform(X_test)

#%% Data2 "Title"
data2 = prep.data_title

y2 = data2["status"]
X2 = data2.drop(columns=["status","subject"])

X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.3, random_state=42)

Xv2_train = vectorize.fit_transform(X2_train)
Xv2_test=vectorize.transform(X2_test)

#%% LR

lr=LogisticRegression()
def LR(X_train, y_train, X_test, y_test): #send vectorized
    lr.fit(X_train,y_train)
    pred = lr.predict(X_test)
    score = lr.score(X_test, y_test)
    return pred, score

#%% Plot for heatmap of confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
def heatmap_cm(clf,X, y, acc, name):
    plot_confusion_matrix(clf, X, y)
    plt.title('{0} Accuracy Score: {1}'.format(name,acc), size = 15)
    print("check")
    plt.show()

#%%

text_pred, text_score = LR(Xv_train, y_train, Xv_test, y_test)
title_pred, title_score = LR(Xv2_train, y_train, Xv2_test, y_test)


heatmap_cm(lr, Xv_test, y_test, text_score, "Logistic Regression Text")
heatmap_cm(lr, Xv2_test, y_test, title_score, "Logistic Regression Title")
