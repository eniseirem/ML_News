import matplotlib.pyplot as plt
import preprocess as prep

data = prep.data_text

print(data.groupby(['status'])['text'].count())
data.groupby(['status'])['text'].count().plot(kind="bar")
plt.show()


print(data.groupby(['subject'])['text'].count())
data.groupby(['subject'])['text'].count().plot(kind="bar")
plt.show()


