"""
This is a machine learning project utilizing Decision Tree and SVM classification methods to group two large
numbers of observations into classes. In this case, following datasets to be classified are used:
    - a sonar mine/rock recognition dataset (https://machinelearningmastery.com/standard-machine-learning-datasets/)
    - an optical recognition of handwritten digits dataset (one of the standard scikit-learn datasets)

To run this project make sure that you:
    - download Python 3
    - download and install numpy with the following command via the terminal: pip install numpy,
    - download and install skfuzyy with the following command via the terminal: pip install scikit-learn
    - and launch the project from your favorite IDE or with the following command: python main.py

    Project created by:
        Kajetan Welc
        Daniel Wirzba
"""

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import svm, datasets

# Loading data from sonar.csv file containing sonar returns at different angles with classes (M for mine and R for rock)
sonar = np.genfromtxt('sonar.csv', delimiter=',', dtype=str)

sonar_X = sonar[:, :-1].astype(float)
sonar_y = sonar[:, -1]

sonar_X_train, sonar_X_test, sonar_y_train, sonar_y_test = train_test_split(
    sonar_X, sonar_y, test_size=0.5)


sonar_dc_classifier = DecisionTreeClassifier(random_state=0, max_depth=100)
sonar_dc_classifier.fit(sonar_X_train, sonar_y_train)

# Predicting classification outcomes with the Decision Tree classification method
sonar_y_test_dc_pred = sonar_dc_classifier.predict(sonar_X_test)

print("Sonar | Decision Tree | Training Report")
print(classification_report(sonar_y_train,
      sonar_dc_classifier.predict(sonar_X_train)))
print()

print("Sonar | Decision Tree | Testing Report")
print(classification_report(sonar_y_test, sonar_y_test_dc_pred))
print()

# Predicting classification outcomes with the SVM classification method
sonar_svm_classifier = svm.SVC()
sonar_svm_classifier.fit(sonar_X_train, sonar_y_train)

sonar_y_test_svm_pred = sonar_svm_classifier.predict(sonar_X_test)

print("Sonar | SVM | Training Report")
print(classification_report(sonar_y_train,
      sonar_svm_classifier.predict(sonar_X_train)))
print()

print("Sonar | SVM | Testing Report")
print(classification_report(sonar_y_test, sonar_y_test_svm_pred))
print()

# Loading data from the standard handwritten digits dataset that already comes with scikit-learn
digits = datasets.load_digits()

digits_X = digits.data
digits_y = digits.target

digits_X_train, digits_X_test, digits_y_train, digits_y_test = train_test_split(
    digits_X, digits_y, test_size=0.5)


digits_dc_classifier = DecisionTreeClassifier(random_state=0, max_depth=100)
digits_dc_classifier.fit(digits_X_train, digits_y_train)

# Predicting classification outcomes with the Decision Tree classification method
digits_y_test_dc_pred = digits_dc_classifier.predict(digits_X_test)

print("Digits | Decision Tree | Training Report")
print(classification_report(digits_y_train,
                            digits_dc_classifier.predict(digits_X_train)))
print()

print("Digits | Decision Tree | Testing Report")
print(classification_report(digits_y_test, digits_y_test_dc_pred))
print()

# Predicting classification outcomes with the SVM classification method
digits_svm_classifier = svm.SVC()
digits_svm_classifier.fit(digits_X_train, digits_y_train)

digits_y_test_svm_pred = digits_svm_classifier.predict(digits_X_test)

print("Digits | SVM | Training Report")
print(classification_report(digits_y_train,
      digits_svm_classifier.predict(digits_X_train)))
print()

print("Digits | SVM | Testing Report")
print(classification_report(digits_y_test, digits_y_test_svm_pred))
print()
