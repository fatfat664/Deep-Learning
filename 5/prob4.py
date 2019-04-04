import torch
import torchvision
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Using the pretrained model
alexnet = torchvision.models.alexnet(pretrained=True)
alexnet.eval()

# Defining transfomrmation to tensor
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# Path for the images
dir = "./data/"
testset = torchvision.datasets.ImageFolder(dir, transform=transform)
data_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

for i, data in enumerate(data_loader):
    inputs, labels = data

    # extracting layer 1 output
    layer1 = alexnet.features[0](inputs).squeeze(0)
    layer1_size = layer1.size()

    # 'Plotting' the output from the first layer for each image
    channel1 = layer1[2].detach().numpy()
    plt.imshow(channel1)
    plt.show()

    # Final layer Output
    output = alexnet(inputs)
    output_size = output.size()
    
    # Testing classification probabilities on test images
    score, class_id = torch.topk(output, dim=1, k=5) # Getting the top 5 labels
    scores = class_id.view(-1).numpy()
    
    print(scores)
    #class_probabilities = scores/all_scores.sum()
    #print("Class Probabilties = {}".format(class_probabilities))
        
print("Shape of the first layer = {}".format(layer1_size))
print("Shape of the outer layer = {}".format(output_size))