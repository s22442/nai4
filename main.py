from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import svm

sonar = np.genfromtxt('sonar.csv', delimiter=',', dtype=str)

sonar_X = sonar[:, :-1].astype(float)
sonar_y = sonar[:, -1]

sonar_X_train, sonar_X_test, sonar_y_train, sonar_y_test = train_test_split(
    sonar_X, sonar_y, test_size=0.5)


sonar_dc_classifier = DecisionTreeClassifier(random_state=0, max_depth=8)
sonar_dc_classifier.fit(sonar_X_train, sonar_y_train)

sonar_y_test_dc_pred = sonar_dc_classifier.predict(sonar_X_test)

print("Sonar | Decision Tree | Training Report")
print(classification_report(sonar_y_train,
      sonar_dc_classifier.predict(sonar_X_train)))
print()

print("Sonar | Decision Tree | Testing Report")
print(classification_report(sonar_y_test, sonar_y_test_dc_pred))
print()

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
