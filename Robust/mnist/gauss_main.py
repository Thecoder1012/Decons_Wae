import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
import os
import torchvision
import sys

random.seed(800)
np.random.seed(800)
torch.manual_seed(800)
torch.backends.cudnn.deterministic=True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def jenson_shannon_divergence(net_1_logits, net_2_logits):
    net_1_probs = F.softmax(net_1_logits, dim=0)
    net_2_probs = F.softmax(net_2_logits, dim=0)
    
    total_m = 0.5 * (net_1_probs + net_1_probs)
    
    loss = 0.0
    loss += F.kl_div(F.log_softmax(net_1_logits, dim=0), total_m, reduction="batchmean") 
    loss += F.kl_div(F.log_softmax(net_2_logits, dim=0), total_m, reduction="batchmean") 
    return (0.5 * loss)

#Converting data to torch.FloatTensor
transform = torchvision.transforms.ToTensor()

# Download the training and test datasets
train_data = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform)

test_data = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, num_workers=2)
print(len(test_data))
print(len(train_data))

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()


def imshow(img):
    #img = img / 2 + 0.5  
    fig = plt.figure(figsize = (5,5))
    ax = fig.add_subplot(111)
    #print(img.shape)
    img = np.transpose(img, (1, 2, 0))
    ax.imshow(img, cmap='gray')
    #plt.imshow(np.transpose(img, (1, 2, 0)))

dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

print(images[1].shape)

input_size = 1*28*28

class Encoder(nn.Module):
  def __init__(self):
    super().__init__()

    #conv layer
    self.enc_1 = torch.nn.Linear(input_size, 512, bias = False)
    
    self.new_enc3 = torch.nn.Linear(512, 256, bias = False)
    self.new_enc4 = torch.nn.Linear(256, 128)
    self.enc_2 = torch.nn.Linear(128, 64)
    

  def forward(self, x):
    
    x = x.reshape(x.size(0), input_size)
    
    x = F.relu(self.enc_1(x))
    
    x = F.relu(self.new_enc3(x))
    x = F.relu(self.new_enc4(x))
    x = F.relu(self.enc_2(x))
    
    return x

class Decoder(nn.Module):
  def __init__(self):
    super().__init__()

    #hidden layer
    
    self.dec_3 = torch.nn.Linear(64, 128)
    self.new_dec1 = torch.nn.Linear(128, 256)
    self.new_dec2 = torch.nn.Linear(256, 512, bias = False)
    
    self.dec_4 = torch.nn.Linear(512, input_size, bias = False)

    #conv layer
    
    

  def forward(self, x):
    
    x = F.relu(self.dec_3(x))
    x = F.relu(self.new_dec1(x))
    x = F.relu(self.new_dec2(x))
    
    x = torch.sigmoid(self.dec_4(x))
    
    x = x.reshape(x.size(0), 1,28,28)
   
    return x

def add_noise(inputs,noise_factor=0.3):
  noisy = .7 * inputs + torch.randn_like(inputs) * noise_factor
  noisy = torch.clip(noisy,0.,1.)
  return noisy

size_of_batch = 1000
rec_loss = []
wae_mmd_loss = []
tot_loss = []
ITERS = 200  #100000
n_z = 64
print('latent space',n_z)
sigma = 1
step = 0
epoch = 0
lambda_per = 1
rec_loss = []
wae_jsd_loss = []
total_loss = []

encoder, decoder = Encoder().to(device), Decoder().to(device)
pytorch_total_params = sum(p.numel() for p in encoder.parameters())
print(pytorch_total_params)
pytorch_total_params = sum(p.numel() for p in decoder.parameters())
print(pytorch_total_params)
print(encoder)
print(decoder)
criterion = nn.MSELoss()

criterion.to(device)

encoder.to(device)
decoder.to(device)

encoder.train()
decoder.train()

enc_optim = optim.Adam(encoder.parameters(), lr= 0.0001)
dec_optim = optim.Adam(decoder.parameters(), lr= 0.0001)

enc_scheduler = StepLR(enc_optim, step_size=30, gamma=0.5)
dec_scheduler = StepLR(dec_optim, step_size=30, gamma=0.5)
batch_size = 120
noise_per = 0.0
image_per = 1.0
path = "sampling_v2_new_"+str(noise_per)+"_"+str(image_per)+"_bs_"+str(batch_size)+"ls_"+str(n_z)+"ac_relu"
os.mkdir(path)
original_stdout = sys.stdout
cnt = 0


for epoch in range(ITERS):
  trcnt = 0
  for data in train_loader:
    images, _ = data
    #images_noisy = add_noise(images)
    if trcnt == 0:
      a = torch.randn_like(images) * noise_per
    if trcnt < 5 :
      images_noisy = a + image_per * images
    else:
      images_noisy = images

    images = images.to(device)
    images_noisy = images_noisy.to(device)
    
    #print(images.shape)
    x_latent = encoder(images_noisy).to(device)
    #print(x_latent.shape)
    #break
    x_recon = decoder(x_latent).to(device)
    #print(x_recon.shape)
    #x_recon = model(images)

    recon_loss = criterion(x_recon, images).to(device)
    p = Variable(torch.rand(images.size()[0], n_z) * sigma).to(device)
    js_loss = jenson_shannon_divergence(p,x_latent).to(device)

    total_loss = recon_loss + lambda_per * js_loss
    
    enc_optim.zero_grad()
    dec_optim.zero_grad()
    total_loss.backward()
    enc_optim.step()
    dec_optim.step()
        
  if (epoch+1) % 10 == 0:
    count = 0
    cnt = -1
    for data in test_loader:
      images, _ = data
      if count % 3 :
        images_noisy = a + images
      else:
        images_noisy = images
      count = count + 1
      #images_noisy = a
      images = images.to(device)
      images_noisy = images_noisy.to(device)

      encoder.eval()
      decoder.eval()
      x_latent = encoder(images_noisy).to(device)
      x_recon = decoder(x_latent).to(device)
      images_noisy = images_noisy.cpu().numpy()

      output = x_recon.view(images_noisy.shape[0], 1, 28, 28)
      output = output.cpu().detach().numpy()
      encoder.train()
      decoder.train()
      cnt = cnt+1
      if epoch + 1 == ITERS and count < 15:
        fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))
        for images_noisy, row in zip([images_noisy, output], axes):
          for img, ax in zip(images_noisy, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.savefig(path+'/'+str(epoch+1)+'_'+str(cnt+1)+'plots.png')

    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))
    for images_noisy, row in zip([images_noisy, output], axes):
      for img, ax in zip(images_noisy, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig(path+'/'+str(epoch+1)+'plots.png')

    

  rec_loss.append(recon_loss.data.item())
  wae_jsd_loss.append(js_loss.data.item())
  tot_loss.append(total_loss.data.item())
  if (epoch+1) % 10 == 0:
    print("Epoch: [%d/%d], Reconstruction Loss: %.6f JS Loss: %.6f Total Loss: %.6f" %(epoch + 1, ITERS, recon_loss.data.item(), js_loss.item(), total_loss))
  

dataiter = iter(test_loader)
images, labels = data
#Sample outputs
encoder = encoder.to(device)
decoder = decoder.to(device)
images = images.to(device)

output_l = encoder(images)
output = decoder(output_l)
images = images.cpu().numpy()

output = output.view(images.shape[0], 1, 28, 28)
output = output.cpu().detach().numpy()

# path = "bs_"+str(batch_size)+"ls_"+str(n_z)+"ac_relu"
# os.mkdir(path)
# original_stdout = sys.stdout

with open(path+'/model_structure.txt', 'w') as fp:
    sys.stdout = fp
    print("activation function: ReLu, bias = False, Lambda = 1")
    print(encoder)
    print(decoder)
    sys.stdout = original_stdout

print(output.shape)
print(images.shape)
# input images on top row, reconstructions on bottom
for images, row in zip([images, output], axes):
  for img, ax in zip(images, row):
      ax.imshow(np.squeeze(img), cmap='gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      plt.savefig(path+'/plots.png')

for images, row in zip([images[10:20], output[10:20]], axes):
  for img, ax in zip(images, row):
      ax.imshow(np.squeeze(img), cmap='gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      plt.savefig(path+'/plots1.png')
torch.save(torch.tensor(rec_loss), path+'/rec_loss.pth')
torch.save(torch.tensor(wae_jsd_loss), path+'/js_loss.pth')
torch.save(torch.tensor(tot_loss), path+'/total_loss.pth')
