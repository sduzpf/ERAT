#import torchvision
import torch
from subprocess import call
from torchvision import datasets

if __name__ == '__main__':
    #torchvision.datasets.MNIST(root='data', download=True)
    
    #cifar-10
    datasets.CIFAR10(root='data/cifar10', train=True, download=True)#training set
    datasets.CIFAR10(root='data/cifar10', train=False, download=True)#test set
    
    #cifar-100
    #datasets.CIFAR100(root=train_data_path, train=True, download=True)
    #datasets.CIFAR10(root=train_data_path, train=False, download=True)