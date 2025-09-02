import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt

col_names = ['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age','label']
pima = pd.read_csv('diabetes_pima.csv',names=col_names,header=None)
print(pima.head())

feature_cols = ['pregnant','insulin','bmi','age','glucose','bp','pedigree']
X = pima[feature_cols]
y = pima.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=6,splitter='best',criterion='entropy')
clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
print(metrics.classification_report(y_test,y_pred))
print(metrics.confusion_matrix(y_test,y_pred))

#wykres drzewa decyzyjnego
plt.figure(figsize=(16,8))
plot_tree(clf,
          feature_names=feature_cols,
          class_names = ["No diabets","Diabetes"],
          filled=True,rounded = True)
plt.title("Decision Tree")
plt.savefig("drzewo.png")
plt.show()
