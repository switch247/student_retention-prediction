
import pickle
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import seaborn as sns
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

student = pd.read_csv('dataset.csv')

# check the shape of the dataset in student DataFrame
print(student.shape)

# How the data looks
print(student.sample(4))
print(student.head(5))

# Check info about all the columns
print(student.info())

print(student.isnull().sum())
print(student.duplicated().sum())

print(student['Target'].unique())

student['Target'] = student['Target'].map({
    'Dropout': 0,
    'Enrolled': 1,
    'Graduate': 2
})

# Check Target column, it must have filled with 0, 1 & 2
print(student)

# Target column is integer now
student.dtypes

# Learn the data mathematically
student.describe()

# Finally find the correlation of Target with all other numeric columns
student.corr()['Target']

# Looking at the corelation, we need to select the required columns for prediction.

# This is the new Df considering relevant input and output columns
student_df = student.iloc[:, [1, 11, 13, 14, 15, 16, 17, 20, 22, 23, 26, 28, 29, 34]]

student_df.head()
student_df.info()

sns.heatmap(student_df)

# How many dropouts, enrolled & graduates are there in Target column
student_df['Target'].value_counts()

# Extract Input & Output Columns
X = student_df.iloc[:, 0:13]
y = student_df.iloc[:, -1]
print(X)

# Splitting the data into Training & Testing Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# Random Forest Classifier

clf = RandomForestClassifier(max_depth=10, random_state=0)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Without Scaling and without CV: ", accuracy_score(y_test, y_pred))
scores = cross_val_score(clf, X_train, y_train, cv=10)
print("Without Scaling and With CV: ", scores.mean())

clf = RandomForestClassifier(bootstrap=False, max_depth=10, max_features=3,  min_samples_split=12,n_estimators=100, random_state=0)
clf.fit(X_train, y_train)




y_pred = clf.predict(X_test)
print("Without CV: ", accuracy_score(y_test, y_pred))
scores = cross_val_score(clf, X_train, y_train, cv=10)
print("With CV: ", scores.mean())

print("Precision Score: ", precision_score(y_test, y_pred, average='micro'))
print("Recall Score: ", recall_score(y_test, y_pred, average='micro'))
print("F1 Score: ", f1_score(y_test, y_pred, average='micro'))

x = {0:'Dropout', 1:'Enrolled', 2:'Graduate'}
for i in range(len(y_pred)):
    print(f"Actual: {x[y_test.values[i]]}, Predicted: {x[y_pred[i]]}")

# Save the model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(clf, file)

