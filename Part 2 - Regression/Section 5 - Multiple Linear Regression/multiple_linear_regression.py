import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# Importing data set and setting independent/dependent variables
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

# Encoding categorical data
# --Encodes country category from string to integer
labelendcoder_X = LabelEncoder()
X[:, 3] = labelendcoder_X.fit_transform(X[:, 3])
# --Transforms country integers to 3 "true or false" columns for each country
# --This is because there is no hierarchy to the countries, they're simply categories
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting dataset into Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fitting Multiple Linear Regression to the training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
# Adding constant to the model
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

# # Starting the manual elimination process. Step 1 - Gather all independent variables and set significance level (0.05)
# X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# # Step 2 - Fit regressor to model
# regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
# # Step 3 - Find predictor with highest p-value
# regressor_OLS.summary()
# # Step 4 - Remove if above Significance Level (p-value of column 2 was 0.99, way above 0.05)
# X_opt = X[:, [0, 1, 3, 4, 5]]
# # Step 5 - re-fit the model and GO AGANE
# regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
# regressor_OLS.summary()
# # p-value of 1 was 0.94
# X_opt = X[:, [0, 3, 4, 5]]
# regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
# regressor_OLS.summary()
# # p-value of 4 was 0.602
# X_opt = X[:, [0, 3, 5]]
# regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
# regressor_OLS.summary()
# # p-value of 5 was 0.06
# X_opt = X[:, [0, 3]]
# regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
# regressor_OLS.summary()
# # p-value of 3 is 0.000 - Model is complete


# Automatic backward elimination
def backward_elimination(x, sl):
    numvars = len(x[0])
    for i in range(0, numvars):
        regressor_ols = sm.OLS(y, x).fit()
        maxvar = max(regressor_ols.pvalues).astype(float)
        if maxvar > sl:
            for j in range(0, numvars - i):
                if regressor_ols.pvalues[j].astype(float) == maxvar:
                    x = np.delete(x, j, 1)
    regressor_ols.summary()
    return x


SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backward_elimination(X_opt, SL)
