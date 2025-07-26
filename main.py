# # Importing essential libraries
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Reading the dataset
# df = pd.read_csv('height-weight.csv')  # Make sure the CSV file is in the same directory or provide full path

# # Optional: Show first few rows
# # print(df.head())

# # Scatter plot to visualize relationship
# # plt.scatter(df['Height'], df['Weight'])
# # plt.title('Height vs Weight')
# # plt.xlabel('Height (inches)')
# # plt.ylabel('Weight (pounds)')
# # plt.show()

# # Correlation matrix
# # print(df.corr())

# # Optional: Pairplot with regression lines
# # sns.pairplot(df, kind='reg')
# # plt.show()

# # Feature selection
# X = df[['Weight']]  # Independent variable (should be a DataFrame)
# y = df['Height']    # Dependent variable (should be a Series)

# # Splitting the dataset into training and test sets (75% train, 25% test)
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.25, random_state=42)

# # Standardizing the features using StandardScaler
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()

# # Fit on training data and transform both training and test data
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Linear Regression model training
# from sklearn.linear_model import LinearRegression
# regression = LinearRegression()
# regression.fit(X_train, y_train)

# # Printing learned parameters
# print("Intercept:", regression.intercept_)
# print("Coefficient:", regression.coef_)

# # Plotting the regression line with training data
# plt.scatter(X_train, y_train, color='blue', label='Training data')
# plt.plot(X_train, regression.predict(X_train), color='red', label='Regression Line')
# plt.title("Linear Regression: Training Set")
# plt.xlabel("Standardized Weight")
# plt.ylabel("Height")
# plt.legend()
# plt.show()

# # Making predictions on the test set
# y_pred = regression.predict(X_test)

# # Evaluating the model
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# mse = mean_squared_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("Mean Absolute Error:", mae)
# print("Mean Squared Error:", mse)
# print("R-squared:", r2)

# # Plotting the regression line with training data again (optional)
# plt.scatter(X_train, y_train, color='blue', label='Training data')
# plt.plot(X_train, regression.predict(X_train), color='red', label='Regression Line')
# plt.title("Final Regression Line")
# plt.xlabel("Standardized Weight")
# plt.ylabel("Height")
# plt.legend()
# plt.show()


# # ✅ If you want model summary, use statsmodels like this (optional):

# import statsmodels.api as sm
# X_const = sm.add_constant(X)  # Adds intercept term
# model = sm.OLS(y, X_const).fit()
# print(model.summary())


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df_index=pd.read_csv('economic_index.csv')

df_index.drop(columns=["No.","year","month"],inplace=True)  # Drop the 'No.' column
# print(df_index.head())
# print(df_index.isnull().sum())  # Check for missing values
# sns.pairplot(df_index)
df_index.corr()
plt.scatter(df_index['interest_rate'], df_index['unemployment_rate'],color='red', label='Data Points')
# plt.show()
X=df_index.iloc[:, :-1]  # All columns except the last one
y=df_index.iloc[:, -1]  # Last column
# print(X.head(),y.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
sns.regplot(x='interest_rate', y='unemployment_rate', data=df_index, color='blue')
# plt.show()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score
Validation_score = cross_val_score(regression, X_train, y_train, scoring='neg_mean_squared_error', cv=3)
print(np.mean(Validation_score))

y_pred = regression.predict(X_test)
#performance metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
mse=mean_squared_error(y_test, regression.predict(X_test))
plt.scatter(y_test, y_pred, color='blue', label='Test Data')
# plt.show()
residuals = y_test - y_pred
print(residuals)
sns.displot(residuals, kde=True)
plt.scatter(y_test, residuals, color='green', label='Residuals')
plt.show()
print(regression.intercept_, regression.coef_)//