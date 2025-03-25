import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
iris = datasets.load_iris()
x = iris.data[:, 2].reshape(-1, 1) # Using petal length as feature
y = (iris.target == 2).astype(int) # Binary classification (Iris Virginica or not)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Virginica',
'Virginica'], yticklabels=['Not Virginica', 'Virginica'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
x_range = np.linspace(x.min(), x.max(), 1000).reshape(-1, 1)
x_range_scaled = scaler.transform(x_range)
y_pred_curve = model.predict_proba(x_range_scaled)[:, 1]
plt.figure(figsize=(6, 4))
plt.scatter(x, y, edgecolors='k', facecolors='none', label='Actual Data')
plt.plot(x_range, y_pred_curve, color='green', linewidth=2, label='Logistic Curve')
plt.xlabel('Petal Length')
plt.ylabel('Probability of Virginica')
plt.title('Logistic Regression Curve')
plt.legend()
plt.show()
print(classification_report(y_test, y_pred))
