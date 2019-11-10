import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.preprocessing import normalize, scale
from matplotlib import style


def organise_data():
    data = pd.read_csv("Titanic.csv")
    Survived = data["Survived"]
    # print(data.info())

    new_data = data.drop(["PassengerId", "Name", "Ticket", "Cabin", "Survived"], axis=1)
    new_data["Embarked"].replace(['C', 'Q', 'S'], ["0", "1", "2"], inplace=True)
    new_data["Embarked"] = new_data["Embarked"].astype(np.float)
    new_data["Sex"].replace(["male", "female"], ["0", "1"], inplace=True)
    new_data["Sex"] = new_data["Sex"].astype(np.float)
    mean_age = (np.mean(new_data["Age"]))
    # print(new_data.info())
    new_data["Age"].fillna(value=mean_age, axis=0, inplace=True)
    new_data["Embarked"].fillna(value=2, axis=0, inplace=True)

    bias = np.ones((new_data.shape[0], 1))
    new_data.insert(0, "X0", bias, True)

    y = np.array(Survived)
    x = np.array(new_data)
    # print(x, y)
    # print(new_data.dtypes)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3, random_state=24)
    x_test = scale(x_test)
    x_test = normalize(x_test)
    x_train = scale(x_train)
    x_train = normalize(x_train)
    style.use("ggplot")

    m = x_train.shape[0]
    features = x_train.shape[1]
    theta = np.zeros((1, m))

    iterations = 1000
    alpha = 1

    for i in range(iterations):
        cost_here = 0
        der_cost = np.zeros((features, 1))
        hyp = hypothesis(theta, x_train)
        cost_here = cost_function(hyp, y_train, m)
        der_cost = derivative(hyp, x_train, y_train, features)
        theta = theta - (alpha/m) * der_cost
        plt.scatter(i, cost_here)

    predict = np.zeros((m, 1))
    hyp = hypothesis(theta, x_train)
    for i in range(m):
        if hyp[i] >= 0.5:
            predict[i] = 1
        else:
            predict[i] = 0

    acc = 0
    for i in range(m):
        if predict[i] == Survived[i]:
            acc = acc + 1
    print(acc/m)

    n = x_test.shape[0]
    acc_test = 0
    predict1 = np.zeros((n, 1))

    hyp1 = hypothesis(theta, x_test)
    for i in range(0, n):
        if hyp1[i] >= 0.5:
            predict1[i] = 1
        else:
            predict1[i] = 0

    for i in range(0, n):
        if predict1[i] == y_test[i]:
            acc_test = acc_test + 1
    print(acc_test / n)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def hypothesis(theta, x):
    z = np.dot(theta, x)
    return sigmoid(z)


def cost_function(hyp, y, m):
    return sum((-1/m) * (y * np.log(hyp) + (1-y) * np.log(1-hyp)))


def derivative(hyp, x_train, y_train, features):
    der = sum((hyp - y_train) * x_train)
    return der.reshape(features, 1)


if __name__ == '__main__':
    organise_data()