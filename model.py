import torch
import torch.nn as nn
import torch.nn.functional as F
from groupsort import GroupSort

class Encoder(nn.Module):
  def __init__(self):
    super().__init__()

    self.enc_1 = torch.nn.Linear(3,3)
    self.enc_2 = torch.nn.Linear(3, 3)
    self.enc_3 = torch.nn.Linear(3, 3)
    self.enc_4 = torch.nn.Linear(3, 2)
    self.gp = GroupSort(1, axis=1)

  def forward(self, x, gps = 0):
    if gps == 1:
        x = self.gp(self.enc_1(x))
        x = self.gp(self.enc_2(x))
        x = self.gp(self.enc_3(x))
        x = self.gp(self.enc_4(x))

    elif gps == 0:
        x = F.leaky_relu(self.enc_1(x))
        x = F.leaky_relu(self.enc_2(x))
        x = F.leaky_relu(self.enc_3(x))
        x = F.leaky_relu(self.enc_4(x))

    return x

class Decoder(nn.Module):
  def __init__(self):
    super().__init__()

    self.dec_1 = torch.nn.Linear(2, 3)
    self.dec_2 = torch.nn.Linear(3, 3)
    self.dec_3 = torch.nn.Linear(3, 3)
    self.dec_4 = torch.nn.Linear(3, 3)
    self.gp = GroupSort(1, axis=1)

  def forward(self, x, gps = 0):
    if gps == 1:
        x = self.gp(self.dec_1(x))
        x = self.gp(self.dec_2(x))
        x = self.gp(self.dec_3(x))
        x = self.dec_4(x)
    
    elif gps == 0:
        x = F.leaky_relu(self.dec_1(x))
        x = F.leaky_relu(self.dec_2(x))
        x = F.leaky_relu(self.dec_3(x))
        x = self.dec_4(x)
    
    return x

class Encoder_Mnist(nn.Module):
  def __init__(self):
    super().__init__()

    self.enc_1 = torch.nn.Linear(1*28*28, 512, bias = False)
    self.new_enc3 = torch.nn.Linear(512, 256, bias = False)
    self.new_enc4 = torch.nn.Linear(256, 128)
    self.enc_2 = torch.nn.Linear(128, 64)
    self.gp = GroupSort(1, axis=1)

  def forward(self, x, gps = 0):
    if gps == 1:
        x = x.reshape(x.size(0), 1*28*28)
        x = self.gp(self.enc_1(x))
        x = self.gp(self.new_enc3(x))
        x = self.gp(self.new_enc4(x))
        x = self.gp(self.enc_2(x))

    elif gps == 0:
        x = x.reshape(x.size(0), 1*28*28)
        x = F.relu(self.enc_1(x))
        x = F.relu(self.new_enc3(x))
        x = F.relu(self.new_enc4(x))
        x = F.relu(self.enc_2(x))


    return x

class Decoder_Mnist(nn.Module):
  def __init__(self):
    super().__init__()

    #hidden layer
    self.dec_3 = torch.nn.Linear(64, 128)
    self.new_dec1 = torch.nn.Linear(128, 256)
    self.new_dec2 = torch.nn.Linear(256, 512, bias = False)
    self.dec_4 = torch.nn.Linear(512, 1*28*28, bias = False)
    self.gp = GroupSort(1, axis=1)
    
  def forward(self, x, gps = 0):
    if gps == 1:
        x = self.gp(self.dec_3(x))
        x = self.gp(self.new_dec1(x))
        x = self.gp(self.new_dec2(x))
        x = torch.sigmoid(self.dec_4(x))
        x = x.reshape(x.size(0), 1,28,28)

    elif gps == 0:
        x = F.relu(self.dec_3(x))
        x = F.relu(self.new_dec1(x))
        x = F.relu(self.new_dec2(x))
        x = torch.sigmoid(self.dec_4(x))
        x = x.reshape(x.size(0), 1,28,28)
    
    return x

