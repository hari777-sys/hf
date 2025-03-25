import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)
lr_model = LinearRegression().fit(x_train, y_train)
y_pred = lr_model.predict(x_test)
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}, R²: {r2_score(y_test,y_pred):.2f}")
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Perfect Fit')
plt.xlabel("Actual"), plt.ylabel("Predicted"), plt.title("Actual vs Predicted"),
plt.legend()
plt.show()
