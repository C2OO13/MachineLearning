import matplotlib.pyplot as plt
import numpy as np
from numpy import *


def compute_error(slope, intercept, points):
    error = 0.0
    for i in range(0, len(points)):
        error += (points[i][1] - ((points[i][0] * slope) + intercept)) ** 2
    return error / float(len(points))


def step_gradient(initial_c, initial_m, learning_rate, points):
    c = float(0)
    m = float(0)
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i][0]
        y = points[i][1]
        # c += (x * 999999999 + y*99999999) * (c + 1)
        # print(c)
        c += (y - (initial_m * x + initial_c))
        m += x * (y - (initial_m * x + initial_c))

    c *= -2.0 / N
    m *= -2.0 / N
    c = initial_c - (learning_rate * c)
    m = initial_m - (learning_rate * m)
    return [c, m]


def gradient_descent(initial_slope, initial_intercept, iterations, points, learning_rate):
    m = initial_slope
    c = initial_intercept
    cost = np.zeros(iterations)
    for i in range(0, iterations):
        [c, m] = step_gradient(c, m, learning_rate, points)
        cost[i] = compute_error(m, c, points)
    return [c, m, cost]


def run():
    print()
    print("Numpy version:")
    print("\t" + np.__version__)

    # reading data from file
    points = genfromtxt('data.csv', delimiter=',')
    print(points)
    # defining parameters
    initial_slope = 0.0
    initial_intercept = 0.0
    learning_rate = 0.0003
    iterations = 8700

    print("\nInitial slope is {0} and Initial intercept is {1}.".format(initial_slope, initial_intercept))
    print("Learning rate is {0} and Number of iterations to be performed are {1}.".format(learning_rate, iterations))
    print("Initial error = {0}".format(compute_error(initial_slope, initial_intercept, points)))
    [final_slope, final_intercept, cost] = gradient_descent(initial_slope, initial_intercept, iterations, points,
                                                            learning_rate)
    print("Final error = {0}".format(compute_error(final_slope, final_intercept, points)))

    print("Value of m is {0} and c is {1}".format(final_slope, final_intercept))
    plt.plot(range(iterations), cost)
    plt.show()


if __name__ == '__main__':
    run()
