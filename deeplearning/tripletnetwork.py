import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

##################################################################################
# LSTM
##################################################################################

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, embedding_size):
        super(LSTMModel, self).__init__()
        self.l1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, embedding_size)

    def forward(self, x):
        r_out, (h_n, h_c) = self.l1(x, None) #None represents zero initial hidden state
        out = self.out(r_out[:, -1, :])
        return out

class LSTMNet(nn.Module):
    def __init__(self, embedding_size):
        super(LSTMNet, self).__init__()
        self.lstm_model = LSTMModel(2048, 512, 1, embedding_size)
    
    def forward(self, x):
        out = self.lstm_model(x)
        return out

##################################################################################
# Fused LSTM
##################################################################################

class FusedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, embedding_size):
        super(FusedLSTMModel, self).__init__()
        self.l1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size+512, embedding_size)

    def forward(self, x, x_c3d):
        r_out, (h_n, h_c) = self.l1(x, None) #None represents zero initial hidden state
        y = torch.cat((r_out[:, -1, :], x_c3d), 1)
        out = self.out(y)
        return out

class FusedLSTMNet(nn.Module):
    def __init__(self, embedding_size):
        super(FusedLSTMNet, self).__init__()
        self.lstm_model = FusedLSTMModel(2048, 512, 1, embedding_size)
    
    def forward(self, x, x_c3d):
        out = self.lstm_model(x, x_c3d)
        return out

##################################################################################
# Neural Networks ------>
##################################################################################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 12, 5)
        # self.conv3 = nn.Conv2d(16, 40, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1092, 512)
        self.fc2 = nn.Linear(512, 128)
    
    def forward(self, x):
        x = x.view(1, 64, 40)
        x = x.unsqueeze(0)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        out = self.fc2(x)
        return out

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # self.conv1 = nn.Conv2d(1, 6, 5)
        # self.conv2 = nn.Conv2d(6, 12, 5)
        # # self.conv3 = nn.Conv2d(16, 40, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2_drop = nn.Dropout2d()
        
        self.fc1 = nn.Linear(5888, 1024)
        # self.fc1 = nn.Linear(2304, 800)
        self.fc2 = nn.Linear(1024, 256)
        # self.fc2 = nn.Linear(800, 256)
    
    def forward(self, x):
        # x = x.view(1, 64, 40)
        # x = x.unsqueeze(0)
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        out = self.fc2(x)
        return out


class SimilarityNet(nn.Module):
    def __init__(self):
        super(SimilarityNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 80, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(80, 48, 5)
        self.conv3 = nn.Conv2d(48, 16, 5)
        self.fc1 = nn.Linear(64, 8)
        self.fc2 = nn.Linear(8, 1)
    
    def forward(self, x, y):
        z = torch.stack([x,y], 0)
        z = z.view(2, 64, 40)
        z = z.unsqueeze(0)
        z = self.pool(F.tanh(self.conv1(z)))
        z = self.pool(F.tanh(self.conv2(z)))
        z = self.pool(F.tanh(self.conv3(z)))
        z = z.view(1, -1)
        z = F.relu(self.fc1(z))
        out = self.fc2(z)
        return out


class Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z, x_c3d=None, y_c3d=None, z_c3d=None, similarity="softmax"):
        if x_c3d is not None and y_c3d is not None and z_c3d is not None:
            embedded_x = self.embeddingnet(x, x_c3d)
            embedded_y = self.embeddingnet(y, y_c3d)
            embedded_z = self.embeddingnet(z, z_c3d)
        else:
            embedded_x = self.embeddingnet(x)
            embedded_y = self.embeddingnet(y)
            embedded_z = self.embeddingnet(z)
        
        # pdist = nn.PairwiseDistance(2)
        # embedded_x = embedded_x.unsqueeze(0)
        # embedded_y = embedded_y.unsqueeze(0)
        # embedded_z = embedded_z.unsqueeze(0)
        # dist_a = pdist(embedded_x, embedded_y)
        # dist_b = pdist(embedded_x, embedded_z)

        if similarity=="cosine":
            cos = nn.CosineSimilarity(dim=0, eps=1e-6)
            sim_a = cos(embedded_x, embedded_y)
            sim_b = cos(embedded_x, embedded_z)

        ### SOFTMAX
        if similarity=="softmax":
            eps = 0.000001
            s1 = torch.exp(-torch.norm(embedded_x-embedded_y+eps))
            s2 = torch.exp(-torch.norm(embedded_x-embedded_z+eps))
            sim_a = s1/(s1+s2)
            sim_b = s2/(s1+s2)

        if similarity=="distance":
            pdist = nn.PairwiseDistance(p=2)
            sim_a = pdist(embedded_x.unsqueeze(0), embedded_y.unsqueeze(0))
            sim_b = pdist(embedded_x.unsqueeze(0), embedded_z.unsqueeze(0))

        # gamma = 0.670
        # eps = 0.000001
        # sim_a = torch.exp(-torch.norm(embedded_x-embedded_y+eps)**2/((gamma**2)*4))
        # sim_b = torch.exp(-torch.norm(embedded_x-embedded_z+eps)**2/((gamma**2)*4))

        # sim_a = self.similaritynet(x, y)
        # sim_b = self.similaritynet(x, z)

        return sim_a, sim_b, embedded_x, embedded_y, embedded_z
