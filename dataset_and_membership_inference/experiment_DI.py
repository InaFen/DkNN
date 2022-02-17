from models import cifar10_cnn, cifar10_resnet50
from get_data_refactored import get_data
import tensorflow as tf
import os
import subprocess
import sys
from models_torch import Net
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from os import path
from cleverhans_dataset_inference.src.generate_features_MIA_IF import feature_extractor_MIA

NUM_DATA_POINTS = 10000 # how many of member/non member elements are used
PATH_MODEL = "/home/inafen/jupyter_notebooks/dataset_inference/model_resnet50"

os.environ["CUDA_VISIBLE_DEVICES"] = ""

"""
#get member data
members, testloader = get_data('cifar10', augmentation=False, batch_size=NUM_DATA_POINTS, indices_to_use=range(0, 25000))
iterator_members = members.next()  #TODO iterator_members good name choice?
member_data = iterator_members[0]
member_labels = iterator_members[1]
"""
#torch code source: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

#get train and test data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

#load model or build and save model
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
PATH_MODEL_TORCH = "/home/inafen/jupyter_notebooks/dataset_inference/model_torch.pth"

if path.exists(PATH_MODEL_TORCH):
    net = Net()
    net.load_state_dict(torch.load(PATH_MODEL_TORCH))
else:
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    torch.save(net.state_dict(), PATH_MODEL_TORCH)
net.eval()

#get test accuracy
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
"""
try:
    model = tf.keras.models.load_model(PATH_MODEL)
except:
    model = cifar10_cnn()
    # compile the model
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics="accuracy",
    )
    # train the model
    # if you want specify batch size, learning rates etc.
    history = model.fit(member_data, member_labels, epochs=50, batch_size=64)
    # export model
    model.save(PATH_MODEL)
#train_accuracy = model.evaluate(member_data, member_labels)
"""
#TODO different amount of samples, loops, see e-Mail

#generate features
feature_extractor_MIA(train_loader= trainloader, test_loader= testloader, student= net)

#subprocess.run(['which', 'python'])
#subprocess.run(['/home/inafen/.conda/envs/mia2/bin/python', 'train.py', '--batch_size', '4', '--epochs', '2'], cwd='cleverhans_dataset_inference/src/')
#subprocess.run(['/home/inafen/.conda/envs/mia2/bin/python', 'generate_features.py'], cwd='cleverhans_dataset_inference/src/')

#notebook
