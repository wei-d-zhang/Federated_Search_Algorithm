#!/usr/bin/python
# -*- coding: UTF-8 -*-
import copy
import math
import glob
import scipy.io as sio
import argparse
from torch.utils.data import Dataset, DataLoader, random_split
from matplotlib import pyplot as plt
import csv
import random
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch import optim
from torch.autograd._functions import tensor
from torch.nn import init
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from collections import OrderedDict
import numpy as np
from data_process import Widar_Dataset,data_reader,cifar_10_load,data_reader1,SVHN_load
from model import cifar10_MLP,cifar10_LeNet,fashion_MLP,fashion_LeNet,fashion_ResNet,Widar_LeNet,Widar_ResNet18,Widar_MLP,SVHN_MLP,SVHN_ResNet18,SVHN_LeNet
from resnet import cifar10_ResNet18

client_n = 5  # num of clients
max_comunication = 50  # communication rounds
root = './' 

batch_size = 128 
train_loader, dataset_label, dataset_label_client, train_dataset_client = [], [], [], []
#fedavg_loss = []
fedavg_accuracy = []
CLIENT_EPOCHS = 1 #local training interaion

#### PSO parameter
omega=0.7
c1=2
c2=2




for i in range(client_n):
    j = []
    train_loader.append(j)
    
train_dataset = datasets.FashionMNIST(root="./FashionMNIST/", train=True, transform=transforms.ToTensor(), download=True)

# Define a custom data set for distributing data to clients
class ClientDataset(Dataset):
    def __init__(self, dataset, client_indices):
        self.dataset = dataset
        self.client_indices = client_indices

    def __len__(self):
        return len(self.client_indices)

    def __getitem__(self, index):
        return self.dataset[self.client_indices[index]]

# Split FashionMNIST Data Integration 10 subsets (one subset per client)
data_split = random_split(train_dataset, [len(train_dataset) // client_n] * client_n)

# Creates a list of client_n client datasets
client_datasets = [ClientDataset(train_dataset, split.indices) for split in data_split]

# Creates a list of client_n client data loaders
train_loader = [DataLoader(client_dataset, batch_size=batch_size, shuffle=True) for client_dataset in client_datasets]

test_dataset = torchvision.datasets.FashionMNIST(root='./FashionMNIST/', train=False, transform=torchvision.transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)


current_model = fashion_LeNet()


class FL(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = current_model.to(self.device)
        self.global_best_model = current_model.to(self.device) #Global optimal location
        
        self.local_models = self.get_models() ##Local current location
        self.local_best_models = self.get_models() ###Local optimal location
        
        # Initialize velocities with random values
        self.velocities = [torch.randn_like(param) / 5 - 0.10 for param in self.model.parameters()]
        
        self.local_best_scores = [float('inf')] * client_n
        self.global_best_score = float('inf')  # Initialize with positive infinity
        
    
    def get_models(self):
        models = []
        client_model = copy.deepcopy(self.model)
        for i in range(client_n):
            models.append(client_model)
        return models


    ##### Update the client location
    def local_update(self,i):
        step_model = copy.deepcopy(self.local_models[i])  #Local current location of the i-th client
        local_best_model = copy.deepcopy(self.local_best_models[i])  ### Gets the optimal location for the i th client
        new_parameters = []
      
        #PSO speed update and position update
        for j, (param, lb_param, gb_param) in enumerate(zip(step_model.parameters(), local_best_model.parameters(), self.global_best_model.parameters())):
            #print(step_model.parameters()[j])
            new_v = omega * self.velocities[j]
            new_v = new_v + c1 * random.random() * (lb_param - param)
            new_v = new_v + c2 * random.random() * (gb_param - param)
            self.velocities[j] = new_v
            new_parameters.append(param + self.velocities[j])
            
        # Apply the new parameters to the model
        with torch.no_grad():
            for j, param in enumerate(step_model.parameters()):
                param.data = new_parameters[j].data
                
        self.local_models[i].load_state_dict(step_model.state_dict())  ######Update current location
        #####Calculation of fitness (loss)
        now_param = self.train(train_loader[i],step_model.state_dict())
        now_loss,acc = self.test(now_param)
        #######Search for the optimal local location
        if now_loss <= self.local_best_scores[i]:
            self.local_best_models[i].load_state_dict(now_param) 
            self.local_best_scores[i] = now_loss
            #print("best local loss: ",self.local_best_scores[i])
        
    def run(self):
        comunication_n = 0
        ######Global iterative procedure
        while comunication_n < max_comunication:  # communication number 
            for i in range(client_n):
                self.local_update(i) #####Client local location update and optimal location update
                ########Determine the optimal global location
                if self.local_best_scores[i] <= self.global_best_score:
                    self.global_best_score = self.local_best_scores[i]
                    self.global_best_model = copy.deepcopy(self.local_best_models[i])
            global_loss, global_acc= self.test(self.global_best_model.state_dict())
            fedavg_accuracy.append(round(global_acc, 2))
            print("The loss of the {} round is: {} and the acc is: {}".format(comunication_n,global_loss,global_acc))
            
            comunication_n = comunication_n +1

        print('Over!')
        
        #f = open('results/FedPSO/0.5/fashion-lenet-2.csv', 'w', encoding='utf-8', newline='')
        #csv_write = csv.writer(f)
        #csv_write.writerow(fedavg_accuracy) 
        
    def train(self, t_dataset, model_para_client):
        model_param = model_para_client
        model = self.model
        model.load_state_dict(model_param)
        model.train()  # Set this parameter to trainning mode
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.005)  # Initializes the optimizer
        
        for i in range(CLIENT_EPOCHS):
            for batch_idx, (data, target) in enumerate(t_dataset):
                data = data.to(self.device)
                target = target.to(self.device)
                data, target = Variable(data), Variable(target)  # Convert the data to Variable
                optimizer.zero_grad()  # Optimizer gradient is initialized to zero
                output = model(data)  # Input data into the network and get output, that is, forward propagation
                loss = criterion(output,target)
                loss.backward()  # Back propagation gradient
                optimizer.step()  # Update parameters after finishing a prepass + backpass
        model_state = copy.deepcopy(model.state_dict())
        return model_state
    
    def test(self,model_parameter):
        model_parame = model_parameter
        test_model = self.model  
        test_model.load_state_dict(model_parame)
        test_model.eval()
        
        test_loss = 0  # The initial test loss value is 0
        correct = 0  # Initialize the number of correctly predicted data to be 0
        
        for data, target in test_loader:
            data = data.to(self.device)
            target = target.to(self.device)
            data, target = Variable(data), Variable(target)  
            output = self.model(data)
            _,predicts = torch.max(output.data, 1)
            #confusion.update(predicts.cpu().numpy(), target.cpu().numpy())
                
            test_loss += F.cross_entropy(output, target,
                                             size_average=False).item()  # sum up batch loss Indicates that all loss values are added
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # Add up the number of data that predicted correctly
           
        test_loss /= len(test_loader.dataset)  # Since all loss values are added up, the average loss is obtained by dividing by the total data length
        acc = (100. * correct / len(test_loader.dataset)).tolist()
        
        return test_loss,acc

def main():
    fl = FL()
    fl.run()


if __name__ == "__main__":
    main()
