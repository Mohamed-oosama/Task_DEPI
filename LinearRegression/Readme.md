# Linear_Regression_Guide.py
"""
📊 Linear Regression Guide
هذا الملف يحتوي على شرح كامل لكل مفاهيم Linear Regression
مع أمثلة عملية ورسومات توضيحية مباشرة من الكود
"""

# ------------------------------
# 1️⃣ استيراد المكتبات
# ------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set(style="whitegrid")

# ------------------------------
# 2️⃣ بيانات بسيطة لتوضيح Linear Regression
# ------------------------------
# شرح:
# لدينا مجموعة نقاط (X, y) ونريد إيجاد خط يمر بينها يقلل الخطأ
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([2,4,6,8,10])

# إنشاء نموذج Linear Regression
model = LinearRegression()
model.fit(X, y)

# توقع القيم
y_pred = model.predict(X)

# عرض الميل و intercept
print("Slope (m):", model.coef_)
print("Intercept (b):", model.intercept_)

# رسم النتائج
plt.figure(figsize=(6,4))
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Example')
plt.legend()
plt.show()

# ------------------------------
# 3️⃣ تأثير Outliers
# ------------------------------
# شرح:
# Linear Regression حساس للقيم الشاذة، نقطة بعيدة يمكن أن تغير الخط بشكل كبير
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
plt.show()

# ------------------------------
# 4️⃣ Gradient Descent مبسط
# ------------------------------
# شرح:
# فكرة Gradient Descent: نحسب الخطأ ونعدل الميل و intercept تدريجيًا
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
plt.show()
print(f"Final Slope: {m:.2f}, Intercept: {b:.2f}")

# ------------------------------
# 5️⃣ Multiple Linear Regression
# ------------------------------
# شرح:
# عندما يكون لدينا أكثر من Feature، تصبح المعادلة:
# y = m1*x1 + m2*x2 + ... + mn*xn + b
X_multi = np.array([[1, 50], [2, 60], [3, 65], [4, 70], [5, 80]])  # مساحة + عدد غرف
y_multi = np.array([200, 250, 300, 350, 400])  # سعر المنزل

model_multi = LinearRegression()
model_multi.fit(X_multi, y_multi)
y_multi_pred = model_multi.predict(X_multi)

print("Coefficients (m1, m2):", model_multi.coef_)
print("Intercept (b):", model_multi.intercept_)

# رسم أول Feature مقابل y
plt.figure(figsize=(6,4))
plt.scatter(X_multi[:,0], y_multi, color='blue', label='Actual')
plt.plot(X_multi[:,0], y_multi_pred, color='red', label='Predicted')
plt.xlabel('Feature 1 (Area)')
plt.ylabel('Price')
plt.title('Multiple Linear Regression Example')
plt.legend()
plt.show()

# ------------------------------
# 6️⃣ Polynomial Regression
# ------------------------------
# شرح:
# عندما تكون العلاقة غير خطية، نستخدم Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures

X_poly = np.array([[1],[2],[3],[4],[5]])
y_poly = np.array([1, 4, 9, 16, 25])  # علاقة x^2

poly = PolynomialFeatures(degree=2)
X_poly_transformed = poly.fit_transform(X_poly)

model_poly = LinearRegression()
model_poly.fit(X_poly_transformed, y_poly)
y_poly_pred = model_poly.predict(X_poly_transformed)

plt.figure(figsize=(6,4))
plt.scatter(X_poly, y_poly, color='blue', label='Actual')
plt.plot(X_poly, y_poly_pred, color='red', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression Example')
plt.legend()
plt.show()
