# Artificial NEural Network

# Step 1: Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Deal with categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Encoding Country
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
#Encoding Genders
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2: Making a ANN
# Importing libraries
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# Initializing ANN - sequence of layers
classifier = Sequential()
#
## Adding the input layer and the first hidden layer with dropout (deals with overfitting)
classifier.add(Dense(units=6,kernel_initializer='uniform',activation = 'relu', input_dim=11))
classifier.add(Dropout(rate = 0.1))
## Adding the second hidden layer with dropout
classifier.add(Dense(units=6,kernel_initializer='uniform',activation = 'relu'))
classifier.add(Dropout(rate=0.1))
## Adding the output layer
classifier.add(Dense(units=1,kernel_initializer='uniform',activation = 'sigmoid'))
#
## Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#
## Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
#
## Part 3: Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
#
## Make a single prediction
new_prediction = classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction = (new_prediction>0.5)
#
## Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Part 4: Tuning, evaluating and improving the ANN

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6,kernel_initializer='uniform' , activation = 'relu', input_dim=11))
    classifier.add(Dense(units=6,kernel_initializer='uniform' , activation = 'relu'))
    classifier.add(Dense(units = 1,kernel_initializer='uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout Regularization to reduce overfitting, if needed (see variance)

# Tuning he ANN
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import cross_val_score

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6,kernel_initializer='uniform' , activation = 'relu', input_dim=11))
    classifier.add(Dense(units=6,kernel_initializer='uniform' , activation = 'relu'))
    classifier.add(Dense(units = 1,kernel_initializer='uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size' : [25, 32],
              'epochs' : [100, 500],
              'optimizer' : ['adam', 'rmsprop'] }

grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print(best_parameters)
print(best_accuracy)

# Best Parameters = 32, 500, adam
# Best Accuracy = 84,5 %
