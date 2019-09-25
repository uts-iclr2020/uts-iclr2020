import torch
from torch import nn, optim
import torch.autograd as autograd
from torch.nn import functional as F

from scipy import optimize
import numpy as np

class TS(nn.Module):

    def __init__(self):
        super().__init__()
        self.temperature = torch.Tensor([1.0])
        
    def find_best_T(self, logits, labels):      
        nll_criterion = nn.CrossEntropyLoss()

        def eval(temperature):
            self.temperature = torch.Tensor(temperature)
            L = logits/self.temperature
            loss = nll_criterion(L, labels)
            return loss
        
        minimum = optimize.fmin(eval, self.temperature, disp=False)
        self.temperature = minimum[0]
        
        return self.temperature.item()

class UTS(nn.Module):

    def __init__(self):
        super().__init__()
        self.t = 1.0
        self.w = 1.0
        
    def find_best_T(self, logits, labels):      
        
        nll_criterion = nn.CrossEntropyLoss()
        
        #Variables:
        nb_samples = logits.size(0)
        nb_classes = logits.size(1)
        self.q_y = torch.ones(nb_classes)
        for k in range(nb_classes):
            self.q_y[k] = labels[labels==k].size(0)/nb_samples
            
        self.confidence = F.softmax(logits , dim=1)   
            
        def eval_w(w):
            loss = self.w_loss(logits, w)            
            return loss

        def eval_t(t):   
            loss = self.t_loss(logits, t)
            return loss

        minimum_w = optimize.fmin(eval_w, self.w, disp=False)
        self.w = minimum_w[0]

        minimum_t = optimize.fmin(eval_t, self.t, disp=False)
        self.t = minimum_t[0]

        return self.t
        
    def weight_function(self, logits, w):
        r''' Weight function '''
        
        w = torch.Tensor([w]) 
        
        x = 1/(torch.exp((1/w)*torch.log((1/logits+1e-12)-1)))        
        return x

    def w_loss(self, logits, w):
        r''' Loss function to optimize T'''
        
        loss = 0        
        nb_classes = logits.size(1)
        nb_samples = logits.size(0)
        for k in range(nb_classes):

            index_predicted_k = (self.confidence.max(1)[1] == k)
            index_predicted_not_k = (self.confidence.max(1)[1] != k)
            confidence_class_k_for_predicted_k  = self.confidence[index_predicted_k, k]+1e-12
            confidence_class_k_predicted_not_k = self.confidence[index_predicted_not_k, k]+1e-12
            
            nb_samples_class_k = confidence_class_k_for_predicted_k.size(0)
            nb_samples_not_class_k = confidence_class_k_predicted_not_k.size(0)
            
            missing_ratio = 1 - (0.99*nb_samples_class_k/(nb_samples*self.q_y[k]))
            missing_ratio = 0 if missing_ratio < 0 else missing_ratio
  
            loss += ((1/nb_samples)*torch.sum(self.weight_function(confidence_class_k_predicted_not_k, w)) - (missing_ratio))**2                

        return loss

    def t_loss(self, logits, t): 
        r''' Loss function to optimize T''' 

        t = torch.Tensor([t])
        confidence_t = F.softmax(logits/t , dim=1)

        loss = 0
        loss_1 = 0
        loss_2 = 0       
        
        nb_classes = logits.size(1)
        for k in range(nb_classes):

            index_predicted_k = (confidence_t.max(1)[1] == k)
            index_predicted_not_k = (confidence_t.max(1)[1] != k)
            confidence_k_for_predicted_k  = confidence_t[index_predicted_k, k]+1e-12
            confidence_k_predicted_not_k = self.confidence[index_predicted_not_k, k]+1e-12
            confidence_k_predicted_not_k_t = confidence_t[index_predicted_not_k, k]+1e-12
            
            loss_1 -= torch.sum(torch.log(confidence_k_for_predicted_k))
            
            weights = self.weight_function(confidence_k_predicted_not_k, self.w)                
            loss_2 -= torch.sum(weights*torch.log(confidence_k_predicted_not_k_t))  

        loss = loss_1 + (1/(nb_classes-1))*loss_2
        return loss



