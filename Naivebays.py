import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , accuracy_score , classification_report
import warnings
warnings.filterwarnings("ignore")
import pickle

df = pd.read_csv("iris.csv")
df = df.drop(['Id'],axis=1)
df['Species'] = df['Species'].map({'Iris-setosa':0 , 'Iris-versicolor':1 , 'Iris-virginica':2}).astype(int)

X = df.iloc[: , : -1]
y = df.iloc[: , -1]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# give the train data to algorithm
from sklearn.naive_bayes import GaussianNB
reg = GaussianNB()
reg.fit(X_train,y_train)

# Train Performance
ans_on_sat = reg.predict(X_train)
print(confusion_matrix(y_train , ans_on_sat))
print(accuracy_score(y_train , ans_on_sat))
print(classification_report(y_train , ans_on_sat))

print("===========================================")

# Test Performance
ans_on_sun = reg.predict(X_test)
print(confusion_matrix(y_test , ans_on_sun))
print(accuracy_score(y_test , ans_on_sun))
print(classification_report(y_test , ans_on_sun))


# own datapoint prediction

big_leaf_length = 1.2
big_leaf_width = 2.5
small_leaf_length = 3.6
small_leaf_width = 1.9

result = reg.predict([[big_leaf_length , big_leaf_width , small_leaf_length , small_leaf_width]])[0]

print("===========Prediction with own data point=======================")

if result == 0:
    print(f"Given Tree leaf was : Setosa")
elif result == 1:
    print(f"Given Tree leaf was : Versicolor")
else:
    print(f"Given Tree leaf was : Verginica")

with open("Iris_Model_Naive_Bayes.pkl" , "wb") as f:
    pickle.dump(reg , f)


# load and check once
with open("Iris_Model_Naive_Bayes.pkl","rb") as p:
    m = pickle.load(p)

result = m.predict([[big_leaf_length , big_leaf_width , small_leaf_length , small_leaf_width]])[0]

print("===========Prediction with own data point=======================")

if result == 0:
    print(f"Given Tree leaf was : Setosa")
elif result == 1:
    print(f"Given Tree leaf was : Versicolor")
else:
    print(f"Given Tree leaf was : Verginica")
