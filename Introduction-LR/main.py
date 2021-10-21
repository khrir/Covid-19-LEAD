# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score

# Read the data
df = pd.read_csv("covid19.csv")
cdf = df[['last_available_confirmed', 'last_available_deaths']]

# Create a mask
msk = np.random.rand(len(df)) < 0.6
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

# Plotting train model
plt.scatter(train.last_available_confirmed, train.last_available_deaths, color='blue')
plt.plot(train_x, (regr.coef_[0][0] * train_x) + regr.intercept_[0], '-r')
plt.xlabel("Confirmed")
plt.ylabel("Deaths")
plt.title("Train model")
plt.show()

# Evaluation
test_x = np.asanyarray(test[['last_available_confirmed']])
test_y = np.asanyarray(test[['last_available_deaths']])
test_y_ = regr.predict(test_x)

# Plotting test model
plt.scatter(test.last_available_confirmed, test.last_available_deaths, color='blue')
plt.plot(test_x, (regr.coef_[0][0] * test_x) + regr.intercept_[0], '-r')
plt.xlabel("Confirmed")
plt.ylabel("Deaths")
plt.title("Test model")
plt.show()

# Accuracy rate
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares: %.2f" % np.mean((test_y_ - test_y)**2))
print("R2-score: %.5f" % r2_score(test_y_, test_y))
