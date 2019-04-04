import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()  # Loading the data

net = network.Network([784, 10])  # 784 input neurons, 10 output neurons, no hidden layers
net.SGD(training_data, 30, 50, 5.0, test_data=test_data)  # Mini-batch size = 50, Learning rate = 5.0