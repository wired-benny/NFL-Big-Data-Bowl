import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

dt = pd.read_csv('THEdataset.csv', low_memory = False)
dt = dt.replace(np.inf,np.nan)
dt = dt.dropna()
dt = dt.sample(n=200000)

dt = dt.drop(columns=['GameId','PlayId','NflId','NflIdRusher','Deliver','NewDeliver'])

dt['PlayerBirthYear'] = dt['PlayerBirthDate'].apply(lambda x: x.split('/')[2]).astype('int')
dt['PlayerHeightDecimal'] = dt['PlayerHeight'].apply(lambda x: int(x.split('-')[0]) + int(x.split('-')[1])/12)
dt['GameClockMinutes'] = dt['GameClock'].apply(lambda x: int(x.split(':')[1]) + int(x.split(':')[2])/60)

from datetime import datetime
dt['TimeHandoff'] = dt['TimeHandoff'].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
dt['TimeSnap'] = dt['TimeSnap'].apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
dt['PlayDuration'] = dt.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)

dt = dt.drop(columns=['PlayerBirthDate'])
dt = dt.drop(columns=['PlayerHeight'])
dt = dt.drop(columns=['GameClock'])
dt = dt.drop(columns=['TimeSnap'])
dt = dt.drop(columns=['TimeHandoff'])

dt['Deliver'] = (dt['Yards'] >= 10).astype('int')
dt = dt.drop(columns=['Yards'])

dt_with_dummies = pd.get_dummies(dt)

# Splitting the dataset into dependable variable and independent vector
target_column = 'Deliver'
# This will take the list of all the columns you have and will remove the
# target_column one.
feature_columns = list(dt_with_dummies.columns)
feature_columns.remove(target_column)

X = dt_with_dummies[feature_columns]
Y = dt_with_dummies[target_column]

# Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#RandomForest
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0,n_jobs = -1)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)# Making the Confusion Matrix
cmrf = confusion_matrix(Y_test, Y_pred)
accuracyrf = accuracy_score(Y_pred, Y_test)
print(accuracy_score(Y_test,Y_pred))
print(roc_auc_score(Y_test,Y_pred))

# Create an empty, unlearned tree
decision_tree = DecisionTreeClassifier(criterion="entropy")
# Fit/train the tree on the training data
decision_tree.fit(X_train, Y_train)
# Get a prediction from the tree on the test data
Y_pred = decision_tree.predict(X_test)
# Get the accuracy of this prediction
accuracydt = accuracy_score(Y_pred, Y_test)
# Print the accuracy
print("The accuracy is {}".format(accuracydt))
print(roc_auc_score(Y_test,Y_pred))
cmdt = confusion_matrix(Y_test, Y_pred)