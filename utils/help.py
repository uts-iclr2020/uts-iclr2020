import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split

class MeanStdInfo:

    def __init__(self):
        self.n, self.sum, self.sum_squared = (0.0, 0.0, 0.0)
        
    def update(self, x):
        self.n += 1.0
        self.sum += x
        self.sum_squared += x**2
        self.mean = self.sum/self.n
        self.std = np.sqrt(self.sum_squared/self.n - self.sum**2/self.n/self.n)
        
    def __repr__(self):
        return '%.4f' % self.mean + " +/- " + '%.4f' % self.std

    def __str__(self):
        return '%.4f' % self.mean + " +/- " + '%.4f' % self.std

def load_logits_labels(dataset_name, model_name):

    path = os.path.join('logits/', dataset_name, model_name)
    
    logits = torch.load(path + "/logits.pth")
    labels = torch.load(path + "/labels.pth")
    return logits, labels
    
def data_split(logits, labels, test_size):

    idx = np.array(list(range(len(labels))))
    valid, test = train_test_split(idx, test_size=test_size, stratify=labels)

    v_logits = logits[valid]
    v_labels = labels[valid]
    t_logits = logits[test]
    t_labels = labels[test]
    
    return v_logits, v_labels, t_logits, t_labels
