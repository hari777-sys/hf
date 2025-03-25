
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
# Load the Iris dataset
iris = load_iris()
x = iris.data
y = iris.target
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Create a Decision Tree Classifier model
clf = DecisionTreeClassifier(random_state=42)
# Fit the model on the training data
clf.fit(x_train, y_train)
# Predict the target values for the test data
y_pred = clf.predict(X_test)
# Check the accuracy
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
# Plot the decision tree
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, filled=True, feature_names=iris.feature_names,
class_names=iris.target_names, rounded=True)
plt.title("Decision Tree for Iris Dataset")
plt.show()
