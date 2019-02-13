import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch

def create_and_save_data(N=[15, 100], Sigma=[0, 0.05, 0.2]):
    file = open("data.txt","w")

    for num in N:
        for S in Sigma:
            noise = np.random.normal(0, S, num)
            x = np.random.uniform(0, 3, num)
            y = x ** 2 - 3 * x + 1
            y = y + noise

            plt.scatter(x, y)
            description = '{} Points. Noise with 0 mean and Sigma {}'.format(num, S)
            plt.title(description)
            plt.show()
            plt.pause(1)
            # plt.savefig(description+'.png', format='svg', dpi=72)

            file.write("x = " + str(x) + "\n\n")
            file.write("y = " + str(y) + "\n\n\n")

            fit_polynomial_equation(x, y, num, S)

def fit_polynomial_equation(X, Y, num, S, degrees=[1, 2, 9]):
    u = patch.Patch(color='orange', label='Underlying Points')
    one = patch.Patch(color='blue', label='1st Degree')
    two = patch.Patch(color='red', label='2nd Degree')
    nine = patch.Patch(color='green', label='9th Degree')

    i=0
    graphlist = ['orange', 'blue', 'red', 'green']

    for degree in degrees:
        vander_x = np.vander(X, degree + 1)
        xtx = np.dot(vander_x.transpose(), vander_x)
        xtxi = np.linalg.inv(xtx)
        xty = np.dot(vander_x.transpose(), Y)
        coeffs = xtxi.dot(xty)

        new_y = vander_x.dot(coeffs)
        mse = np.sum((Y - new_y) ** 2)
        print('Degree : {} MSE : {} '.format(degree, mse))
        print("Coeffs : ", np.round(coeffs, 2))

        description = '{} Points. Fitting n-degree polynomial equation with Sigma {}'.format(num, S)
        plt.title(description)
        plt.scatter(X, Y, color = graphlist[0])
        plt.scatter(X, new_y, color = graphlist[i+1])
        plt.legend(handles=[u, one, two, nine])
        print()

        i = i + 1
        create_vandermonde_matrix([1.0000, 1.5000, 2.0000, 2.5000, 3.0000], 4)

    plt.show()
    plt.pause(1)

def create_vandermonde_matrix(x, degree):
    x = np.asarray(x)
    vandermonde_matrix = np.empty((x.shape[0], degree + 1), dtype=np.float32)
    vandermonde_matrix[:, degree] = 1
    vandermonde_matrix[:, degree - 1] = x
    for i in range(0, degree - 1):
        vandermonde_matrix[:, i] = x ** (degree - i)

    return vandermonde_matrix

create_and_save_data()