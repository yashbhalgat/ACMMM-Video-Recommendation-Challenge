######################################################################################
# Author: Yash Sanjay Bhalgat
# email id: yashsb@umich.edu
# CVBRP ACMMM Challenge
######################################################################################

from torch import nn
from torch.autograd import Function
import sys 
import time
import numpy as np
import cv2
import torch
import pdb

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, (1./n)**0.5) 

def batchnorm(x):
    return nn.BatchNorm2d(x.size()[1])(x)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=5, bn = False, relu = True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.bn is not None:
            x = self.bn(x)
        return x

class Full(nn.Module):
    def __init__(self, inp_dim, out_dim, bn = False, relu = False):
        super(Full, self).__init__()
        self.fc = nn.Linear(inp_dim, out_dim, bias = True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.fc(x.view(x.size()[0], -1))
        if self.relu is not None:
            x = self.relu(x)
        if self.bn is not None:
            x = self.bn(x)
        return x

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        if self.batch_first:
            x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        else:
            x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class Net(nn.Module):
    def __init__(self, inp_channels=105, increase=16, bn=False):
        super(Net, self).__init__()
        self.inp_channels = inp_channels
        self.increase = increase
        self.pool = nn.MaxPool2d(2, 2)

        # Left
        n_inp_channels = inp_channels + increase
        self.conv1_left = Conv(inp_channels, n_inp_channels, kernel_size=5, bn=bn)

        inp_channels = n_inp_channels
        n_inp_channels = inp_channels + increase
        self.conv2_left = Conv(inp_channels, n_inp_channels, kernel_size=5, bn=bn)

        # Check dimensions!!!
        self.fc1_left = Full(n_inp_channels * 5 * 13, 1024, bn=bn, relu=True)
        self.fc2_left = nn.Linear(1024, 512)

        # Right
        inp_channels = self.inp_channels
        n_inp_channels = inp_channels + increase
        self.conv1_right = Conv(inp_channels, n_inp_channels, kernel_size=5, bn=bn)

        inp_channels = n_inp_channels
        n_inp_channels = inp_channels + increase
        self.conv2_right = Conv(inp_channels, n_inp_channels, kernel_size=5, bn=bn)

        # Check dimensions!!!
        self.fc1_right = Full(n_inp_channels * 5 * 13, 1024, bn=bn, relu=True)
        self.fc2_right = nn.Linear(1024, 512)


    def forward(self, left_video, right_video):
        left_video = left_video.view(-1, 105, 32, 64)
        right_video = right_video.view(-1, 105, 32, 64)

        left = self.pool(self.conv1_left(left_video))
        left = self.pool(self.conv2_left(left))
        left = left.view(-1, (self.inp_channels+2*self.increase) * 5 * 5)
        left = self.fc1_left(left)
        left = F.sigmoid(self.fc2_left(left))

        right = self.pool(self.conv1_right(right_video))
        right = self.pool(self.conv2_right(right))
        right = left.view(-1, (self.inp_channels+2*self.increase) * 5 * 5)
        right = self.fc1_right(right)
        right = F.sigmoid(self.fc2_right(right))

        ## Output dot product?

        return left, right

class LSTM_Net(nn.Module):
    def __init__(self, num_hidden_lstm, inp_dim):
        super(LSTM_Net, self).__init__()
        self.num_hidden_lstm = num_hidden_lstm
        self.input_size = input_size

        self.lstm_left = nn.LSTM(hidden_size=num_hidden_lstm,
                            input_size=input_size,
                            num_layers=1,
                            dropout=0.33,
                            batch_first=True)

        mlp_left = nn.Sequential(nn.Linear(self.num_hidden_lstm, self.input_size, True), nn.Sigmoid())
        self.tdmlp_left = TimeDistributed(mlp)
        
        self.lstm_right = nn.LSTM(hidden_size=num_hidden_lstm,
                            input_size=input_size,
                            num_layers=1,
                            dropout=0.33,
                            batch_first=True)

        mlp_right = nn.Sequential(nn.Linear(self.num_hidden_lstm, self.input_size, True), nn.Sigmoid())
        self.tdmlp_right = TimeDistributed(mlp)

    def forward(self, left_video, right_video):
        left_lstm_out = self.lstm_left(left_video)
        

        right_lstm_out = self.lstm_right(right_video)


if __name__ == '__main__':
    # define device as the first visible cuda device if we have CUDA available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Network
    net = Net(bn=True)

    device = torch.device("cuda:0")
    net.to(device)

    ## Setup data loader