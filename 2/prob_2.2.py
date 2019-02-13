import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def shaping_and_splitting_data(n,k):
    training_error_list = []
    testing_error_list = []

    # Leave out one cross validation loops n times
    if k == -1:
        k = len(data)

    for i in range(0, k):

        shuffled_data = data
        if (k != -1):  # Only shuffle if its not Leave Out One Cross Validation
            np.random.shuffle(shuffled_data)  # Shuffling Data

        split = int(((k - 1) / k) * len(shuffled_data))  # Splitting Data
        train_data = shuffled_data[:split]
        test_data = shuffled_data[split:]

        train_x = train_data[:, :6]
        train_y = train_data[:, 7]
        test_x = test_data[:, :6]
        test_y = test_data[:, 7]

        # Fitting the model
        neigh = NearestNeighbors(n_neighbors = n)
        neigh.fit(train_x, train_y)

        distances, indices = neigh.kneighbors(train_x)

        # Training error
        error = 0
        for input_index, original_class in zip(indices, train_y):
            classes = [train_y[neighbor] for neighbor in input_index]
            majority_class, _ = stats.mode(classes)
            if majority_class != original_class:
                error = error + 1
        training_error = error / len(train_y)
        training_error_list.append(training_error)

        distances, indices = neigh.kneighbors(test_x)

        # Testing error
        error = 0
        for input_index, original_class in zip(indices, test_y):
            classes = [train_y[neighbor] for neighbor in input_index]
            majority_class, _ = stats.mode(classes)
            if majority_class != original_class:
                error = error + 1
        testing_error = error / len(test_y)
        testing_error_list.append(testing_error)

    # Averaging the errors
    avg_training_error = round(np.mean(training_error_list), 3)
    avg_testing_error = round(np.mean(testing_error_list), 3)
    print('Training error = ', avg_training_error)
    print('Testing error = ', avg_testing_error)

    # Checking for overfitting
    if (avg_testing_error > avg_training_error):
        print("Overfitting")
    return avg_testing_error, avg_training_error

def plotting(train, test, kvalues, c):
    patch = [mpatches.Patch(color='red', label='Training Error'), mpatches.Patch(color='blue', label='Testing Error')]

    plt.legend(handles=patch)
    plt.plot(kvalues, test, color='red')
    plt.plot(kvalues, train, color='blue')

    if(c==5):
        plt.title('Error vs k - 5 Fold Cross Validation')
    else:
        plt.title('Error vs k - Leave Out One Cross Validation')
    plt.xlabel('k')
    plt.ylabel('Error')
    plt.show()

if __name__ == '__main__':
    data = np.genfromtxt('data_seed.dat', delimiter=',')

    k_set = [1, 5, 10, 15]

    testing_errors = []
    training_errors = []
    # 5 fold cross validation
    print('\n\n5 Fold Cross Validation')
    for k in k_set: # Looping through K set
        print('\nk = ', k)
        testing_error, training_error = shaping_and_splitting_data(k,5)  # Calulating the average error
        testing_errors.append(testing_error)
        training_errors.append(training_error)

    plotting(training_errors, testing_errors, k_set, 5)  # Plotting k vs errors

    testing_errors = []
    training_errors = []
    # Leave one out validation
    print('\n\nLeave Out One Cross Validation')
    for k in k_set: # Looping through K set
        print('\nk = ', k)
        testing_error, training_error = shaping_and_splitting_data(k, -1)  # Calulating the average error
        testing_errors.append(testing_error)
        training_errors.append(training_error)

    plotting(training_errors, testing_errors, k_set, -1)  # Plotting k vs errors