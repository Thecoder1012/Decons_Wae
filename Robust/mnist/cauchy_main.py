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
from skimage.util import random_noise

random.seed(800)
np.random.seed(800)
torch.manual_seed(800)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class Encoder(nn.Module):
  def __init__(self):
    super().__init__()

    # five hidden layers
    self.enc_1 = torch.nn.Linear(3, 32)
    self.enc_2 = torch.nn.Linear(32, 64)
    self.enc_3 = torch.nn.Linear(64, 32)
    self.enc_4 = torch.nn.Linear(32, 2)

  def forward(self, x):

    x = F.relu(self.enc_1(x))
    x = F.relu(self.enc_2(x))
    x = F.relu(self.enc_3(x))
    x = F.relu(self.enc_4(x))
    return x

class Decoder(nn.Module):
  def __init__(self):
    super().__init__()

    self.dec_1 = torch.nn.Linear(2, 32)
    self.dec_2 = torch.nn.Linear(32, 64)
    self.dec_3 = torch.nn.Linear(64, 32)
    self.dec_4 = torch.nn.Linear(32, 3)

  def forward(self, x):

    x = F.relu(self.dec_1(x))
    x = F.relu(self.dec_2(x))
    x = F.relu(self.dec_3(x))
    x = self.dec_4(x)
    return x

def jenson_shannon_divergence(net_1_logits, net_2_logits):
    net_1_probs = F.softmax(net_1_logits, dim=0)
    net_2_probs = F.softmax(net_2_logits, dim=0)
    
    total_m = 0.5 * (net_1_probs + net_1_probs)
    
    loss = 0.0
    loss += F.kl_div(F.log_softmax(net_1_logits, dim=0), total_m, reduction="batchmean") 
    loss += F.kl_div(F.log_softmax(net_2_logits, dim=0), total_m, reduction="batchmean") 
    return (0.5 * loss)

def imq_kernel(X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int):
    batch_size = X.size(0)

    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / (batch_size)
        stats += res1 - res2

    return stats


#Converting data to torch.FloatTensor
transform = torchvision.transforms.ToTensor()

# Download the training and test datasets
train_data = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform)

test_data = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform)



train_loader = torch.utils.data.DataLoader(train_data, batch_size=1000, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, num_workers=0)

print(len(test_data))
print(len(train_data))

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def imshow(img):
    #img = img / 2 + 0.5  
    fig = plt.figure(figsize = (5,5))
    ax = fig.add_subplot(111)
    #print(img.shape)
    img = np.transpose(img, (1, 2, 0))
    ax.imshow(img, cmap='gray')
    #plt.imshow(np.transpose(img, (1, 2, 0)))


input_size = 1*28*28

class Encoder(nn.Module):
  def __init__(self):
    super().__init__()

    
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
    self.dec_3 = torch.nn.Linear(64, 128)
    self.new_dec1 = torch.nn.Linear(128, 256)
    self.new_dec2 = torch.nn.Linear(256, 512, bias = False)
    
    self.dec_4 = torch.nn.Linear(512, input_size, bias = False)

  def forward(self, x):
    x = F.relu(self.dec_3(x))
    x = F.relu(self.new_dec1(x))
    x = F.relu(self.new_dec2(x))
    
    x = torch.sigmoid(self.dec_4(x))
    
    x = x.reshape(x.size(0), 1,28,28)
    
    return x

def add_noise(inputs,noise_factor=0.3):
  noisy = inputs+torch.randn_like(inputs) * noise_factor
  noisy = torch.clip(noisy,0.,1.)
  return noisy


size_of_batch = 1000
rec_loss = []
wae_mmd_loss = []
tot_loss = []
ITERS = 100  #100000
n_z = 64
print('latent space',n_z)
sigma = 1
step = 0
epoch = 0
lambda_per = 1
rec_loss = []
wae_mmd_loss = []
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
batch_size = 128
path = "robust_test_bs_"+str(batch_size)+"ls_"+str(n_z)+"ac_relu/"
os.mkdir(path)
original_stdout = sys.stdout

n_samples = [1000, 3000, 5000, 7000, 9000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 60000]
for sample_size in n_samples:
    rec_loss_list = []
    mmd_loss_list = []
    total_loss_list = []
    total_models = 20
    for model_num in range(total_models):
        cnt = -1
        subset_indices = random.sample(range(0, len(train_data)), sample_size)
        subset_tar = torch.utils.data.Subset(train_data, subset_indices)
        train_loader_subset_tar = torch.utils.data.DataLoader(subset_tar, batch_size=1000, num_workers=0, shuffle=False)
        #print(type(train_loader_subset))
        #print(type(train_loader_subset_tar))
        encoder, decoder = Encoder().to(device), Decoder().to(device)
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

        for epoch in range(ITERS):
            i = 0
            for data in train_loader_subset_tar:
              images,_ = data
              #print(type(train_loader_subset_tar))
              images_noisy = add_noise(images)
    
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
              
              #mmd loss
              z_fake = Variable(torch.rand(images.size()[0], n_z) * sigma).to(device)


              z_real = encoder(images_noisy).to(device)

              mmd_loss = imq_kernel(z_real.cuda(), z_fake.cuda(), h_dim=2)
              mmd_loss = mmd_loss / batch_size

              total_loss = recon_loss + lambda_per * mmd_loss
              enc_optim.zero_grad()
              dec_optim.zero_grad()
              total_loss.backward()
              enc_optim.step()
              dec_optim.step()

            if epoch+1 == 1 and model_num + 1 ==1:
                #os.makedirs('plots_'+str(sample_size))
                os.makedirs(path+'loss_'+str(sample_size))
                os.makedirs(path+'/loss_'+str(sample_size)+'/rec_loss')
                os.makedirs(path+'/loss_'+str(sample_size)+'/mmd_loss')
                os.makedirs(path+'/loss_'+str(sample_size)+'/total_loss')
                os.makedirs(path+'/loss_'+str(sample_size)+'/encoder')
                os.makedirs(path+'/loss_'+str(sample_size)+'/decoder')
        
            if (epoch+1) == ITERS:
                print("Model Number [%d/%d], Epoch: [%d/%d], Reconstruction Loss: %.6f MMD Loss: %.6f Total Loss: %.6f" %(model_num+1, total_models, epoch + 1, ITERS, recon_loss.data.item(), mmd_loss.item(), total_loss))
                #print("Epoch: [%d/%d], Reconstruction Loss: %.6f Total Loss: %.6f" %(epoch + 1, ITERS, recon_loss.data.item(), total_loss))

        rec_loss_list.append(recon_loss.data.item())
        mmd_loss_list.append(mmd_loss.data.item())
        total_loss_list.append(total_loss.data.item())

    torch.save(torch.tensor(rec_loss_list), path+'loss_'+str(sample_size)+'/rec_loss'+'/rec_'+str(model_num)+'_loss.pth')
    torch.save(torch.tensor(mmd_loss_list), path+'loss_'+str(sample_size)+'/mmd_loss'+'/mmd_'+str(model_num)+'_loss.pth')
    torch.save(torch.tensor(total_loss_list), path+'loss_'+str(sample_size)+'/total_loss'+'/total_'+str(model_num)+'_loss.pth')
    torch.save(encoder.state_dict(),path+'loss_'+str(sample_size)+'/encoder/enc_'+str(model_num)+'.pth')
    torch.save(decoder.state_dict(),path+'loss_'+str(sample_size)+'/decoder/dec_'+str(model_num)+'.pth') 
    print(str(sample_size)+' Done!')
    
    