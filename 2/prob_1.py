import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch

def create_data(N=[15, 100], Sigma = [0.05]):
    file = open('data_2.txt',"w")
    for n in N:
        for s in Sigma:
            noise = np.random.normal(0, s, n)
            x = np.random.uniform(-1, 4, n)
            y = (x**2) -(3*x) + 1
            y = y + noise
            fit_polynomial(x, y, n, s)

def fit_polynomial(x, y, n, s):
    u = patch.Patch(color='black', label='Underlying Points')
    nine = patch.Patch(color='orange', label='9th Degree')

    degree = 9
    lamb = [0.2, 5, 100000000000]
    for i in range(0,len(lamb)):
        vander_x = np.vander(x, degree + 1)
        xtx = np.dot(vander_x.transpose(),vander_x)
        xtxi = np.linalg.inv(xtx)
        xtxixt = np.dot(xtxi, vander_x.transpose())
        optimal_weights = np.dot(xtxixt, y)  # Finding w*

        dimensions = np.shape(xtx)
        regularized_weights = np.dot(np.dot(np.linalg.inv(((lamb[i] * np.identity(dimensions[0])) + xtx)), vander_x.transpose()), y)  # w* = (((Lamb*I)+XTX)inverse).XT.Y

        new_y = np.dot(vander_x, regularized_weights)

        mse = sum((y - new_y)**2)
        print("MSE = {}".format(mse))
        print("Regularized Weights = {}".format(regularized_weights))

        description = '{} Points. Fitting 9-degree poly eq with Sigma={} and Lambda={}'.format(n, s, lamb[i])
        plt.title(description)
        plt.scatter(x, y, color = 'black')
        plt.scatter(x, new_y, color = 'orange')
        plt.legend(handles=[u, nine])
        print()

        plt.show()
        plt.pause(1)

create_data()

