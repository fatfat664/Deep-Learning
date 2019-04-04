import torch
import torchvision
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Parameters
learning_rate = 0.005
momentum = 0.0
num_epochs = 50
batch_size = 32
dropout_rate = 0.5

#Comment this out for L2 regularization
regularization = 'l1'
l1 = 0.9

# To transform to tensor
transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# Dataset for training, validation and test sets as tensors
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms)
mnist_trainset, mnist_validationset = torch.utils.data.random_split(mnist_trainset, [50000, 10000])

num_training_samples = mnist_trainset.__len__()
l1 = (l1 * 1.0) / num_training_samples

# Data loader for train, test and validation sets
trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, num_workers=2, shuffle=True)
testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=batch_size, num_workers=2, shuffle=True)
validationloader = torch.utils.data.DataLoader(mnist_validationset, batch_size=batch_size, num_workers=2, shuffle=True)

# Compute accuracy
def Accuracy(dataLoader):
    total_samples, score = 0, 0
    for i, data in enumerate(dataLoader):
        with torch.no_grad():
            inputs, labels = data
            outputs = model(inputs)
            outputs = torch.argmax(outputs, dim=1)
            correct = sum(outputs == labels).data.to('cpu').numpy()

            total_samples = total_samples + batch_size
            score = score + correct

    accuracy = score * 1.0 / total_samples
    return accuracy


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        layer1_channel = 20
        layer2_channel = 64
        
        self.features = nn.Sequential(
            nn.Conv2d(1, layer1_channel, (5, 5), stride=1, padding=2),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.ReLU(),
            nn.Conv2d(layer1_channel, layer2_channel, (3, 3), stride=1, padding=0),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        
        self.classifier = nn.Linear((36* layer2_channel), 10)

    def forward(self, x):
        x = self.features(x)
        mini_batch_size = x.size(0)

        x = x.view(x.size(mini_batch_size), -1)  # Transforming
        x = self.classifier(x)
        x = F.softmax(x)
        return x

# Getting our model
model = Net()

# Defining optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum) # SGD
loss_func = nn.CrossEntropyLoss() # Cross Entropy

# Starting Training
for epoch in range (0, num_epochs):
    epoch_loss = 0
    for i, data in enumerate(trainloader):
        x, label = data
        inputs, labels = data
        output = model(inputs)

        model.zero_grad()
        loss = loss_func(output, labels)
        
        # For L1 regularization, SGD already performs L2 by default
        if regularization == "l1":
            for param in model.parameters():
                loss = loss + l1 * torch.sum(torch.abs(param)).data.to('cpu').numpy()
        
        loss.backward()
        optimizer.step()
        
        epoch_loss = epoch_loss + loss
        
    print("Epoch {}. Loss/Training Cost = {}".format(epoch, "%.2f" % epoch_loss))

    # Validation accuracy every 10 epochs
    if (epoch) % 10 == 0:
        print("Epoch {}. Validation Accuracy = {}".format(epoch, "%.2f" % Accuracy(validationloader)))

# Test set accuracy
print("Test Accuracy = {}".format("%.2f" % Accuracy(testloader)))