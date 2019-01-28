# Data Processing Template

# Step 1: Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step 2: Importing the dataset and separating variables
dataset = pd.read_csv('Data.csv')
# Matrix of features
X = dataset.iloc[:, :-1].values
# Depedent variable we are studying
y = dataset.iloc[:, -1].values

# Step 3: Working with missing data
# Importing function that deals with missing data
from sklearn.preprocessing import Imputer
# Every numerical variable with NaN becomes the mean of every other variable
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:, 1:3])

# Step 4: Dealing with categorical variables (encoding it)
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
# 1st: Dealing with Countries
# Using dummy variables (no orders between the countries)
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# 2nd: Dealing with Yes or No
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Step 5: Splitting the dataset between training and testing sets
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Step 6: Feature Scaling
# We need to do this since age is varying from 20 to 60 and salary is varying
#from 50000 to 80000
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
train_X = sc_X.fit_transform(train_X)
test_X = sc_X.fit_transform(test_X)