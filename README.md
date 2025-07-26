
# 📊 Economic Index Linear Regression Analysis

This project applies **Linear Regression** to a dataset containing economic indicators such as **interest rate**, **index price**, and **unemployment rate**, with the goal of predicting **unemployment rate**.

---

## 📁 Dataset Description

**Filename:** `economic_index.csv`

**Columns (before cleaning):**

* `No.`: Row index (removed)
* `year`: Year of observation (removed)
* `month`: Month of observation (removed)
* `interest_rate`: The interest rate during that period
* `index_price`: The market index price
* `unemployment_rate`: Target variable — unemployment rate

---

## 🧪 Objective

To:

1. Analyze relationships between variables
2. Predict the unemployment rate using `interest_rate` and `index_price`
3. Evaluate model performance with standard metrics
4. Visualize residuals and fit line

---

## 🛠️ Tools & Libraries

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Statsmodels (optional)

---

## 🧹 Data Preprocessing

* Dropped unnecessary columns: `No.`, `year`, `month`
* Checked for missing values
* Standardized the features using `StandardScaler`

---

## 📈 Model

### 🔍 Algorithm: `LinearRegression` from `sklearn.linear_model`

### 🔁 Cross Validation:

Used **3-fold cross-validation** with **negative MSE** scoring to validate the model on training data:

```python
Validation_score = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=3)
```

---

## 📊 Evaluation Metrics

* **Mean Squared Error (MSE)**
* **Mean Absolute Error (MAE)**
* **R² Score**
* **Residual Analysis & KDE Plot**

### Example Output (may vary per run):

```
Cross-Validation Score (mean MSE): -0.0534
Intercept: 5.24
Coefficients: [-0.42  0.37]
```

---

## 📉 Visualizations

* **Scatter Plot** of:

  * Interest rate vs. Unemployment rate
  * Predicted vs Actual values
* **Residual Plot**:

  * Distribution of residuals (with KDE)
  * Residuals vs Actual values

---

## 📌 Key Observations

* The model attempts to fit a line to a multivariate dataset.
* Residuals help assess whether the model meets assumptions (e.g., homoscedasticity, normality).
* R² and errors indicate how well the model explains variance in unemployment rate.

---

## ✅ Optional Add-ons

* Use `statsmodels.OLS` for a detailed regression summary:

```python
import statsmodels.api as sm
X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()
print(model.summary())
```

---

## 🧾 How to Run

1. Ensure `economic_index.csv` is in the same directory.
2. Run the Python script.
3. Examine the printed metrics and plots.

---

## 📚 Future Improvements

* Explore polynomial regression if non-linearity exists.
* Include time-series analysis if `year` and `month` matter.
* Try Lasso/Ridge for regularization.
