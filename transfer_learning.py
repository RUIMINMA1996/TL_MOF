from __future__ import print_function, division

import torch
import torch.nn as nn 
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable 

import numpy as np 
import matplotlib.pyplot as plt 
import time 
import os 

class NeuralNet(nn.Module):
	def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
		super(NeuralNet, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size_1)
		self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
		self.fc3 = nn.Linear(hidden_size_2, output_size)
		self.relu = nn.ReLU()

	def forward(self, x):
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)
		out = self.relu(out)
		out = self.fc3(out)
		return out 