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
from model import fashion_LeNet
from math import exp

client_n = 5  # num of clients
max_comunication = 50  # communication rounds
root = './' 
batch_size = 128 
train_loader, dataset_label, dataset_label_client, train_dataset_client = [], [], [], []
#fedavg_loss = []
fedavg_accuracy = []
CLIENT_EPOCHS = 1 #local training interaion

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

data_split = random_split(train_dataset, [len(train_dataset) // client_n] * client_n)
client_datasets = [ClientDataset(train_dataset, split.indices) for split in data_split]
train_loader = [DataLoader(client_dataset, batch_size=batch_size, shuffle=True) for client_dataset in client_datasets]
test_dataset = torchvision.datasets.FashionMNIST(root='./FashionMNIST/', train=False, transform=torchvision.transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

model_c = []
current_model = fashion_LeNet()

####Calculate the mass of the particle
def calculateMass(losses):
    mass = []
    Mass = []
    for i in range(len(losses)):
        mass.append((losses[i] - max(losses)) / (min(losses) - max(losses) + 0.0000000000000000000001)) #########Prevent denominator
    for i in range(len(mass)):
        Mass.append(mass[i] / (sum(mass) + 0.00000000000000000000000000001))
    return Mass
  
###Calculate the distance between particle positions
def calculateDistance(param1,param2):
    distance = torch.norm(param1 - param2, p='fro')
    return distance
  
#########Calculate the acceleration of the particle
def calculateAcceleration(models, Mass, G):
    accelerations = []
    F_i = copy.deepcopy(models[0])
    for param in F_i.parameters():
        param.data = torch.zeros_like(param.data)

    for i in range(client_n):
        for j in range(client_n):
            if i != j:
                for param1,param2,F_i_p in zip(models[i].parameters(),models[j].parameters(),F_i.parameters()):
                    F_i_p = F_i_p + random.random() * G * ((Mass[i] * Mass[j]) / (calculateDistance(param1, param2) + 0.000001)) * (
                            param1 - param2)
        
        accelerations.append(F_i) 
        for param in F_i.parameters():
            param.data = torch.zeros_like(param.data)
    return accelerations

class FL(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = current_model.to(self.device)
        self.global_best_model = current_model.to(self.device)
        self.local_models = self.get_models()        ###########Save the model of the particle
        self.losses = [random.uniform(5, 10) for _ in range(client_n)] ###########Initialize the fitness value
        self.velocities = [torch.randn_like(param) / 5 - 0.10 for param in self.model.parameters()]  #######Initialize the velocity of the particle
        #self.velocities = self.initialize_velocities()
        self.global_best_score = float('inf')
        
    def get_models(self):
        models = []
        client_model = fashion_LeNet().to(self.device)
        for i in range(client_n):
            models.append(client_model)
        return models
    
    def initialize_velocities(self):
        velocities = []
        for _ in range(client_n):
            client_velocities = [torch.randn_like(param) / 5 - 0.10 for param in self.model.parameters()]
            velocities.append(client_velocities)
        return velocities

    ##########Update the velocity and position of the particles
    def updateVelocityAndPosition(self, accelerations,i):
        step_model = copy.deepcopy(self.local_models[i])
        new_parameters = []
        for j, (param,acceler) in enumerate(zip(step_model.parameters(),accelerations[i].parameters())):
            new_v = random.random() * self.velocities[j]
            new_v = new_v + acceler
            self.velocities[j] = new_v
            new_parameters.append(param + self.velocities[j])
        with torch.no_grad():
            for j, param in enumerate(step_model.parameters()):
                param.data = new_parameters[j].data
                
        now_param = self.train(train_loader[i],step_model.state_dict())
        self.local_models[i].load_state_dict(now_param) 
        now_loss,acc = self.test(self.local_models[i].state_dict())
        self.losses[i] = now_loss
        
    def run(self):
        comunication_n = 0
        G = 100
        
        while comunication_n < max_comunication:  # communication number 
            
            G = G * exp(-20 * comunication_n / max_comunication) ######### Acceleration coefficient
            Mass = calculateMass(self.losses)
            #print(Mass)
            Accelerations = calculateAcceleration(self.local_models, Mass, G)
            
            for i in range(client_n):
                self.updateVelocityAndPosition(Accelerations,i)
            
            min_loss = float('inf')
            min_index = -1
            for i in range(client_n):
                if self.losses[i] <= min_loss:
                    min_loss = self.losses[i]
                    min_index = i
            self.global_best_score = min_loss
            self.global_best_model = copy.deepcopy(self.local_models[min_index])
            global_loss, global_acc= self.test(self.global_best_model.state_dict())
            fedavg_accuracy.append(round(global_acc, 2))
            print("The loss of the {} round is: {} and the acc is: {}".format(comunication_n,global_loss,global_acc))
            
            comunication_n = comunication_n +1

        print('Over!')
        
        f = open('results/FedGSA/fashion-lenetnet-5.csv', 'w', encoding='utf-8', newline='')
        csv_write = csv.writer(f)
        csv_write.writerow(fedavg_accuracy)
        
    def train(self, t_dataset, model_para_client):
        model_param = model_para_client
        model = self.model
        model.load_state_dict(model_param)
        model.train()  
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.005)  
        
        for i in range(CLIENT_EPOCHS):
            for batch_idx, (data, target) in enumerate(t_dataset):
                data = data.to(self.device)
                target = target.to(self.device)
                data, target = Variable(data), Variable(target) 
                optimizer.zero_grad()  
                output = model(data) 
                loss = criterion(output,target)
                loss.backward()  
                optimizer.step()  
        model_state = copy.deepcopy(model.state_dict())
        return model_state
    
    def test(self,model_parameter):
        model_parame = model_parameter
        test_model = self.model  
        test_model.load_state_dict(model_parame)
        test_model.eval()
        
        test_loss = 0  
        correct = 0  
        
        for data, target in test_loader:
            data = data.to(self.device)
            target = target.to(self.device)
            data, target = Variable(data), Variable(target)  
            output = self.model(data)
            _,predicts = torch.max(output.data, 1)
            #confusion.update(predicts.cpu().numpy(), target.cpu().numpy())
                
            test_loss += F.cross_entropy(output, target,
                                             size_average=False).item()  
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()  
           
        test_loss /= len(test_loader.dataset)  
        acc = (100. * correct / len(test_loader.dataset)).tolist()
    
        return test_loss,acc

def main():
    fl = FL()
    fl.run()


if __name__ == "__main__":
    main()
