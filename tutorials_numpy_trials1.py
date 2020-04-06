import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets

count_df = pd.DataFrame(
    np.array([
        [1,0,1,0,0,0],
        [0,1,0,1,0,0],
        [1,1,1,1,0,0],
        [0,0,0,0,1,1],
        [0,0,0,0,0,1]], dtype='float64'),
    index=['gnarly', 'wicked', 'awesome', 'lame', 'terrible'])

print(count_df)

iris = datasets.load_iris()
X = iris.data
y = iris.target

print(type(X))
print("Dimensions of X:", X.shape)
print(type(y))
print("Dimensions of y:", y.shape)

# split data into train/test
X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(
    X, y, train_size=0.7, test_size=0.3)
print("X_iris_train:", type(X_iris_train))
print("y_iris_train:", type(y_iris_train))
print()

# start up model
maxent = LogisticRegression(
    fit_intercept=True,
    solver='liblinear',
    multi_class='auto')

# train on train set
maxent.fit(X_iris_train, y_iris_train)

# predict on test set
iris_predictions = maxent.predict(X_iris_test)
fnames_iris = iris['feature_names']
tnames_iris = iris['target_names']

# how well did our model do?
print(classification_report(y_iris_test, iris_predictions, target_names=tnames_iris))

from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from scipy import linalg

# cosine distance
a = np.random.random(10)
b = np.random.random(10)
cosine(a, b)

import matplotlib.pyplot as plt
a = np.sort(np.random.random(30))
b = a**2
c = np.log(a)
plt.plot(a, b, label='y = x^2')
plt.plot(a, c, label='y = log(x)')
plt.legend()
plt.title("Some functions")
plt.show()