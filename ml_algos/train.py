 
import numpy as np
from scipy import stats

alpha = 0.05
z_alpha = stats.norm.ppf(alpha)
beta = 0.2
z_beta = stats.norm.ppf(beta)
print(z_alpha, z_beta)

from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTree2 import DecisionTree as DecisionTree2
from DecisionTree import DecisionTree
import pandas as pd
data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)
import datetime
pd.DataFrame(X_train).to_csv("./x_train.csv",index=False)
pd.DataFrame(y_train).to_csv("./y_train.csv",index=False)
pd.DataFrame(X_test).to_csv("./x_test.csv",index=False)
pd.DataFrame(y_test).to_csv("./y_test.csv",index=False)

st = datetime.datetime.now()
clf = DecisionTree2(max_depth=10)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(y_test, predictions)
print("DTOrg", acc, datetime.datetime.now()- st)

st = datetime.datetime.now()

clf = DecisionTree(max_depth=10)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

acc = accuracy(y_test, predictions)

print("My", acc, datetime.datetime.now()- st)


