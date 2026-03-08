# 📊 Linear Regression Guide

هذا الملف يحتوي على شرح شامل لمفهوم **Linear Regression** في تعلم الآلة، مع توضيح أهم المفاهيم المرتبطة به.

---

## 1️⃣ ما هو Linear Regression

**Linear Regression** هو نموذج إحصائي يهدف لإيجاد علاقة خطية بين المتغيرات.  
الفكرة الأساسية: إيجاد أفضل خط مستقيم يمر بين النقاط بحيث يقلل الفرق بين:

- القيم الحقيقية
- القيم المتوقعة من النموذج

المعادلة الأساسية للخط:

\[
y = m \cdot x + b
\]

| الرمز | المعنى |
|-------|--------|
| y     | القيمة المتوقعة |
| x     | المتغير (Feature) |
| m     | الميل (Slope) |
| b     | نقطة التقاطع مع محور y (Intercept) |

---

## 2️⃣ فكرة عمل النموذج

نحاول إيجاد خط يقلل الفرق بين القيمة الحقيقية \(y\) والقيمة المتوقعة \(\hat{y}\).

\[
error = y - \hat{y}
\]

---

## 3️⃣ حساب الميل (Slope) ونقطة التقاطع

**الميل:**

\[
m = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}
\]

**Intercept:**

\[
b = \bar{y} - m \cdot \bar{x}
\]

حيث:

| الرمز | المعنى |
|-------|--------|
| \(x_i\) | قيمة المتغير |
| \(y_i\) | القيمة الحقيقية |
| \(\bar{x}\) | متوسط x |
| \(\bar{y}\) | متوسط y |

---

## 4️⃣ Multiple Linear Regression

عندما يكون لدينا أكثر من Feature:

\[
y = m_1 x_1 + m_2 x_2 + ... + m_n x_n + b
\]

**مثال:** التنبؤ بسعر منزل باستخدام:

- المساحة
- عدد الغرف
- الموقع

---

## 5️⃣ قياس جودة النموذج

### Mean Squared Error (MSE)

\[
MSE = \frac{1}{n} \sum (y - \hat{y})^2
\]

### Root Mean Squared Error (RMSE)

\[
RMSE = \sqrt{MSE}
\]

### معامل التحديد \(R^2\)

\[
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
\]

- 0 → النموذج سيء  
- 1 → النموذج مثالي

---

## 6️⃣ مشاكل شائعة

- **Outliers**: Linear Regression حساس للقيم الشاذة  
- **Overfitting**: عند التعقيد الكبير للنموذج  
- **Multicollinearity**: وجود تداخل بين Features  
- **Non-linear relationships**: عند عدم خطية العلاقة

---

## 7️⃣ Gradient Descent (تعديل الخط تدريجيًا)

\[
b' = b + h \cdot (y - \hat{y})
\]

\[
m' = m + h \cdot x \cdot (y - \hat{y})
\]

حيث \(h\) هو **learning rate**.

---

## 8️⃣ Regularization

لتجنب **Overfitting**:

- **L1 Norm (Lasso Regression)**: اختيار Features مهمة  
\(\|B\|_1 = |B_1| + |B_2| + ... + |B_n|\)

- **L2 Norm (Ridge Regression)**: تقليل معاملات النموذج  
\(\|B\|_2 = \sqrt{B_1^2 + B_2^2 + ... + B_n^2}\)

---

## 9️⃣ Polynomial Regression

عندما تكون العلاقة غير خطية:

\[
y = B_0 + B_1 x + B_2 x^2 + B_3 x^3
\]

**مثال:** منحنى يصف نمو أو تغير معقد بدل خط مستقيم.

---

## 🔟 مقارنة سريعة للنماذج

| النموذج | الاستخدام |
|---------|-----------|
| Linear Regression | علاقة خطية بسيطة |
| Multiple Linear | أكثر من Feature |
| Ridge | تقليل معاملات لتجنب Overfitting |
| Lasso | اختيار Features مهمه |
| Polynomial | علاقة غير خطية |

---

## 1️⃣1️⃣ أهم المشاكل في Linear Regression

- Overfitting  
- Outliers  
- Multicollinearity  
- Non-linear relationships

---

## 📚 المصادر

- [Scikit-Learn Linear Regression](https://scikit-learn.org/stable/modules/linear_model.html)  
- [Machine Learning Mastery](https://machinelearningmastery.com/linear-regression-for-machine-learning/)  
- [Towards Data Science](https://towardsdatascience.com/linear-regression-detailed-view-5a6c1c3a8b59)
