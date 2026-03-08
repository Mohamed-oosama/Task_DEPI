import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

sns.set(style="whitegrid")

# =========================
# 1. بيانات بسيطة للخط المستقيم
# =========================
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([2,4,6,8,10])

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

plt.figure(figsize=(6,4))
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Example')
plt.legend()
plt.savefig('../images/linear_fit.png')
plt.close()

# =========================
# 2. تأثير Outliers
# =========================
X_out = np.array([[1],[2],[3],[4],[5],[6]])
y_out = np.array([2,4,6,8,10,30])  # نقطة شاذة

model.fit(X_out, y_out)
y_out_pred = model.predict(X_out)

plt.figure(figsize=(6,4))
plt.scatter(X_out, y_out, color='blue', label='Actual')
plt.plot(X_out, y_out_pred, color='red', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Effect of Outliers on Linear Regression')
plt.legend()
plt.savefig('../images/outliers.png')
plt.close()

# =========================
# 3. Gradient Descent Visualization
# =========================
# بيانات بسيطة
X_g = np.array([1,2,3,4,5])
y_g = np.array([2,4,6,8,10])

m, b = 0, 0  # البداية
learning_rate = 0.1
iterations = 20

plt.figure(figsize=(6,4))
plt.scatter(X_g, y_g, color='blue', label='Actual')

for _ in range(iterations):
    y_pred_g = m*X_g + b
    plt.plot(X_g, y_pred_g, color='red', alpha=0.3)
    error = y_g - y_pred_g
    m += learning_rate * np.dot(error, X_g) / len(X_g)
    b += learning_rate * error.mean()

plt.title('Gradient Descent Steps')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.savefig('../images/gradient_descent.png')
plt.close()

print("All plots generated in 'images/' folder!")
