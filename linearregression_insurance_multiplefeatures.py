import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def basic():
    print("\t" + np.__version__)
    print("\t" + pd.__version__)
    print("\t" + sns.__version__)
    print(os.listdir("../Machine Learning/"))
    insurance = pd.read_csv("../Machine Learning/insurance.csv")
    print(insurance.head())
    print(insurance.info())
    print(insurance.columns)
    sns.pairplot(insurance)
    num_cols = insurance.select_dtypes(include=np.number).columns
    non_num_cols = insurance.select_dtypes(exclude=np.number).columns
    print("Numeric data " + num_cols)
    print("Non-numeric data " + non_num_cols)
    plt.show()


def step(theta0, theta, learning_rate, features, expenses):
    cons = 0.0
    cost_here = 0.0
    slope = np.zeros((len(theta), 1))
    for j in range(0, len(features)):
        temp = (expenses[j] - (
                cons + np.dot(features[j], theta)))
        print(temp)
        cons += temp
        # slope += (temp * theta)
        cost_here += temp ** 2
    cons *= -2.0 / len(expenses)
    slope *= -2.0 / len(expenses)
    cost_here /= float(len(expenses))
    cons = theta0 - (learning_rate * cons)
    slope = theta - (learning_rate * slope)
    return cons, slope, cost_here


def gradient_descent(features, expenses, theta, theta0, iterations, learning_rate):
    cost = np.zeros(iterations)
    for i in range(0, iterations):
        theta0, theta, cost[i] = step(theta0, theta, learning_rate, features, expenses)
    return cost, theta0, theta


def compute_error(features, expenses, theta, theta0):
    error = 0.0
    for i in range(len(expenses)):
        predict = theta0 + np.dot(features[i], theta)
        error += (expenses[i] - predict) ** 2
    return error / float(len(expenses))


def run():
    insurance = pd.read_csv("../Machine Learning/insurance.csv")
    age = (insurance["age"].copy())
    bmi = (insurance["bmi"].copy())
    children = (insurance["children"].copy())
    expenses = (insurance["expenses"].copy())
    insurance["sex"].replace(["male", "female"], ["1", "0"], inplace=True)
    sex = (insurance["sex"].astype(np.int))
    insurance["smoker"].replace(["yes", "no"], ["1", "0"], inplace=True)
    smoker = (insurance["smoker"].astype(np.int))
    insurance["region"].replace(["southeast", "northwest", "southwest", "northeast"], ["0", "1", "2", "3"],
                                inplace=True)
    region = (insurance["region"].astype(np.int))
    #    print(region)
    #    print(age, bmi, children, expenses, smoker, sex)
    #    print(smoker[0:50])
    iterations = 100
    learning_rate = 0.00003
    features = np.array([age, bmi, children, region, sex, smoker])
    theta0 = 0.0
    theta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # print(len(features.transpose()[0]))
    # print("Initial error is {0}".format(compute_error(features.transpose(), expenses, theta.transpose(), theta0)))
    #    print(len(features[0]))
    # cost, theta0, theta = gradient_descent(features.transpose(), expenses, theta.transpose(), theta0, iterations, learning_rate)
    # print("Final error is {0}".format(compute_error(features.transpose(), expenses, theta.transpose(), theta0)))
    # plt.plot(range(iterations), cost, 'b.')
    # plt.show()


if __name__ == '__main__':
    run()
#    basic()
