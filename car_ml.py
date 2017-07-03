import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import warnings
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
import pickle

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

def get_data(filename):
    try:
        f = open('data.pkl', 'rb')
    except IOError:
        data = pd.read_csv(filename, "r", delimiter=",")
        data.replace("?", np.NaN, inplace=True)
        data = data.dropna()
        # shuffles rows
        data = data.iloc[np.random.permutation(len(data))]
        pickle.dump(data, open('data.pkl', 'wb'))
        return data
    with f:
        return pickle.load(f)

data = get_data('cardata.csv')

def assign_target(column):
    return data[column]
    del data[column] # data now has all other features

target = assign_target('price') # price feature

def clean():
    del data['fuel-system']
    del data['fuel-type']
    del data['aspiration']
    del data['body-style']
    del data['drive-wheels']
    del data['engine-location']
    del data['engine-type']
    data.replace("two", 2, inplace=True)
    data.replace("three", 3, inplace=True)
    data.replace("four", 4, inplace=True)
    data.replace("five", 5, inplace=True)
    data.replace("six", 6, inplace=True)
    data.replace("seven", 7, inplace=True)
    data.replace("eight", 8, inplace=True)

clean()

# hashes car model
data['make'] = data['make'].apply(hash)

# train with this
X_train = data[:-10]
Y_train = target[:-10]

# predict with this
X_test = data[-10:]

#validate with this
Y_test = target[-10:]

def train_linear():
    regr = linear_model.LinearRegression()
    regr.fit(X_train, Y_train)
    print(regr.coef_)
    predicted_Y = regr.predict(X_test)
    # print(predicted_Y, Y_test)
    print(regr.score(X_test, Y_test)) # handles prediction

    # plots actual vs predicted
    plt.scatter(Y_test.index, Y_test, color="black", label="Test Data")
    plt.scatter(Y_test.index, regr.predict(X_test), color="yellow", label="Predicted Test Data")
    print(Y_test, regr.predict(X_test))
    plt.legend()
    plt.show()

# Monday task, predict using polynomials
def train_poly():
    # svr_poly = SVR(kernel="poly", C=1e3, cache_size=7000, degree=3)
    # svr_poly.fit(X_train, Y_train)
    #
    # print(svr_poly.predict(X_test))
    #
    # plt.scatter(Y_test.index, Y_test, color="black", label="Test Data")
    # plt.scatter(Y_test.index, svr_poly.predict(X_test), color="blue", label="Predicted Test Data")
    # print(Y_test, svr_poly.predict(X_test))
    # plt.legend()
    # plt.show()
    poly = PolynomialFeatures(degree=2)
    poly.fit_transform(X_test)
    print(X)
    print(poly.fit_transform(X_test))
train_poly()
# closeness = []
#
#
# for pr in range(len(predicted_Y)):
#     predict = predicted_Y.item(pr)
#     test = Y_test.iloc[pr]
#     closeness.append(abs(float(predict) / float(test)))
#
# print("Closeness: ", closeness)
