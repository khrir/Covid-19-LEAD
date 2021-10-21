# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score

# Read the data
df = pd.read_csv("covid19.csv")
cdf = df[['state', 'last_available_confirmed', 'last_available_deaths']]

# Create a mask
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Linear model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['last_available_confirmed']])
train_y = np.asanyarray(train[['last_available_deaths']])
regr.fit(train_x, train_y)

# Coefficients
print("Coefficient: ", regr.coef_)
print("Intercept: ", regr.intercept_)

# Evaluation
test_x = np.asanyarray(test[['last_available_confirmed']])
test_y = np.asanyarray(test[['last_available_deaths']])
test_y_ = regr.predict(test_x)

print("Residual sum of squares: %.2f" % np.mean((test_y_ - test_y)**2))
print("R2-score: %.5f" % r2_score(test_y_, test_y))

# Plotting
plt.scatter(train.last_available_confirmed, train.last_available_deaths, color='blue')
plt.plot(train_x, regr.coef_[0][0] * train_x + regr.intercept_[0], '-r')
plt.xlabel("Confirmed")
plt.ylabel("Deaths")
plt.show()
