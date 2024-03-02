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
train_loaders = []
fedavg_accuracy = []
CLIENT_EPOCHS = 1 #local training interaion

for i in range(client_n):
    j = []
    train_loaders.append(j)
    
train_dataset = datasets.FashionMNIST(root="./FashionMNIST/", train=True, transform=transforms.ToTensor(), download=True)

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
train_loaders = [DataLoader(client_dataset, batch_size=batch_size, shuffle=True) for client_dataset in client_datasets]
test_dataset = torchvision.datasets.FashionMNIST(root='./FashionMNIST/', train=False, transform=torchvision.transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

model_c = []
current_model = fashion_LeNet()

class FL(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = current_model.to(self.device)
        self.global_best_model = current_model.to(self.device)
        self.local_models = self.get_models()
        self.losses = [random.uniform(5, 10) for _ in range(client_n)]
        self.velocities = [torch.randn_like(param) / 5 - 0.10 for param in self.model.parameters()]
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
    
    def calculateAcceleration(self, t):
        accelerations = []
        for i in range(client_n):
            mean_model = copy.deepcopy(self.local_models[i])
            accele_model = copy.deepcopy(self.local_models[i])
            for paramm,accele in zip(mean_model.parameters(),accele_model.parameters()):
                paramm = torch.zeros_like(paramm)
                accele = torch.zeros_like(accele)
                
            num = 0
            for j in range(client_n):
                if i != j:
                    if self.losses[j] <= self.losses[i]:
                        num = num + 1
                        for param,other_param in zip(mean_model.parameters(),self.local_models[j].parameters()):
                            with torch.no_grad():
                                param += other_param
            for param,accele,param_i in zip(mean_model.parameters(),accele_model.parameters(),self.local_models[i].parameters()):
                param = param / (num + 0.00001)
                p_mean = 50 / (t+1)
                with torch.no_grad():
                    accele = p_mean * param - param_i                                    
            accelerations.append(accele_model)
            
        return accelerations
        
    def updateVelocityAndPosition(self, accelerations,i,t):
        step_model = copy.deepcopy(self.local_models[i])
        new_parameters = []
        k1 = 1/t
        k2 = 1/(1+exp(-t))
        for j, (param,acceler,bestparam) in enumerate(zip(step_model.parameters(),accelerations[i].parameters(),self.global_best_model.parameters())):
            new_v = bestparam - param
            param = param - k1 * random.random() * acceler - k2 *random.random() * new_v
            new_parameters.append(param)
            self.velocities[j] = new_v
        with torch.no_grad():
            for j, param in enumerate(step_model.parameters()):
                param.data = new_parameters[j].data
                
        now_param = self.train(train_loaders[i],step_model.state_dict())
        self.local_models[i].load_state_dict(now_param) 
        now_loss,acc = self.test(self.local_models[i].state_dict())
        self.losses[i] = now_loss
        
    def run(self):
        comunication_n = 1
        
        while comunication_n < max_comunication+1:  # communication number 
            Accelerations = self.calculateAcceleration(comunication_n)
            
            for i in range(client_n):
                self.updateVelocityAndPosition(Accelerations,i,comunication_n)
            
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
        
        f = open('results/FedIPO/fashion-lenetnet-14.csv', 'w', encoding='utf-8', newline='')
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
        #fedavg_loss.append(round(test_loss, 4))
        acc = (100. * correct / len(test_loader.dataset)).tolist()
        #fedavg_accuracy.append(round(acc, 2))
        #print('\n 测试loss: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                                        #len(test_loader.dataset),
                                                                                        #100. * correct / len(test_loader.dataset)))
        return test_loss,acc

def main():
    fl = FL()
    fl.run()



if __name__ == "__main__":
    main()
