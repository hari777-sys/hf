import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
iris = datasets.load_iris()
x = iris.data[:, :2]
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(x_train, y_train)
y_pred = svm_model.predict(x_test)
print(f"Accuracy (Linear Kernel): {accuracy_score(y_test, y_pred):.2f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
def plot_decision_boundary(model, x, y, title):
 h = 0.02
 x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
 y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
 xx,yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
 Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
 Z = Z.reshape(xx.shape)
plt.contourf(xx,yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(title)
plt.show()
plot_decision_boundary(svm_model, X_train, y_train, "SVM Decision Boundary (LinearKernel)")
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)
print(f"\nAccuracy (RBF Kernel): {accuracy_score(y_test, y_pred_rbf):.2f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rbf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rbf))
plot_decision_boundary(svm_rbf, X_train, y_train, "SVM Decision Boundary (RBF Kernel)")
