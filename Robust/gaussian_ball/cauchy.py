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

random.seed(800)
np.random.seed(800)
torch.manual_seed(800)
#torch.use_deterministic_algorithms(True)

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

    x = F.leaky_relu(self.enc_1(x))
    x = F.leaky_relu(self.enc_2(x))
    x = F.leaky_relu(self.enc_3(x))
    x = F.leaky_relu(self.enc_4(x))
    return x

class Decoder(nn.Module):
  def __init__(self):
    super().__init__()

    self.dec_1 = torch.nn.Linear(2, 32)
    self.dec_2 = torch.nn.Linear(32, 64)
    self.dec_3 = torch.nn.Linear(64, 32)
    self.dec_4 = torch.nn.Linear(32, 3)

  def forward(self, x):

    x = F.leaky_relu(self.dec_1(x))
    x = F.leaky_relu(self.dec_2(x))
    x = F.leaky_relu(self.dec_3(x))
    x = self.dec_4(x)
    return x

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


data_pts = 10000
noise_points_list = [2000]

noise_per_list = [0.9]
for n_p in noise_points_list:
  for n_per in noise_per_list:
    print("data_pts:",data_pts)
    print('noise points:',n_p)
    print('noise percentage:',n_per)
    data_pts_load = True
    if data_pts_load == False:
        data_all = inf_train_gen(data_pts)
        torch.save(torch.tensor(data_all),'datapts_sample.pth')
    elif data_pts_load == True:
        data_all = torch.load('datapts_sample.pth').numpy()
    print("actual data length:", len(data_all))
    
    noise_points = data_all[(len(data_all)-n_p):len(data_all)]
    print("length noise pts:",n_p)
    data_noise = data_all[0:(len(data_all)-n_p)]
    data_noise = data_noise.tolist()
    scale = 3
    centers = [
        (0, 0, 0),
        (1, 1, 0),
        (1, -1, 0),
        (1, 0, 1),
        (1 , 0, -1),
        #(-1, 1, 0),
        #(-1, -1, 0 ),
        #(-1, 0, 1),
        #(-1, 0, -1),
    ]
    centers = [(scale * x, scale * y, scale * z ) for x, y, z in centers]
    data_noise1 = []
    dirichlet = np.random.dirichlet((5,3,5), n_p)
    for i in range(n_p):
      cauchy = np.random.standard_cauchy(3) * 0.2
      point = (1 - n_per)* (noise_points[i] * 1.414) + ( n_per * cauchy)
      point[2] = point[2]*random.choice([1,-1])
      center = random.choice(centers)
      point[0] += center[0]
      point[1] += center[1]
      point[2] += center[2]
      data_noise.append(point)
    dataset_noise = np.array(data_noise, dtype='float32')
    dataset_noise /= 1.414 # stdev


    print("noise data length:",len(dataset_noise))
    torch.save(torch.tensor(dataset_noise),'./datasets/datapts_noisedir_'+str(n_p)+'_'+str(n_per)+'.pth')
    dataset_noise = torch.load('./datasets/datapts_noisedir_'+str(n_p)+'_'+str(n_per)+'.pth').numpy()
    data_all = torch.load('datapts_sample.pth').numpy()

    n_samples = [10000]
    for samples in n_samples:
      #n_samples = [50000]
      print("number of samples:",samples)
      epoch_choice = []
      size_of_batch = 1000
      rec_loss = []
      wae_mmd_loss = []
      tot_loss = []
      ITERS = 10000  #100000
      n_z = 2
      sigma = 1
      step = 0
      epoch = 0
      rn = np.random.choice(len(data_all), samples)
      data_inp = dataset_noise[rn]
      data_target = data_all[rn]
      

      encoder, decoder = Encoder(), Decoder()
      pytorch_total_params = sum(p.numel() for p in encoder.parameters())
      
      pytorch_total_params = sum(p.numel() for p in decoder.parameters())
      

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

      if torch.cuda.is_available():
          encoder, decoder = encoder.cuda(), decoder.cuda()
          data_inp = torch.tensor(data_inp)
          ub_list = []
          ub_list = np.arange(0, data_inp.shape[0] + size_of_batch, size_of_batch)
          #print(ub_list)
      t_samples = 2000
      t_samples = int(t_samples)
      rn_test1 = np.random.choice(int(samples),int(t_samples))
        
      for epoch in range(ITERS):
        
        total_steps = data_inp.shape[0] // size_of_batch
        
        for steps in range(data_inp.shape[0] // size_of_batch):
          _data = data_inp[ub_list[steps]:ub_list[steps+1]][:]
          _data_tar = data_target[ub_list[steps]:ub_list[steps+1]][:]
          
          images = autograd.Variable(torch.Tensor(_data)).to(device)
          images_tar = autograd.Variable(torch.Tensor(_data_tar)).to(device)

          enc_optim.zero_grad()
          dec_optim.zero_grad()

          # ======== Train Generator ======== #

          batch_size = images.size()[0]

          z = encoder(images)

          x_recon = decoder(z)


          recon_loss = criterion(x_recon.cuda(), images_tar.cuda())

          # ======== MMD Kernel Loss ======== #

          z_fake = Variable(torch.rand(images.size()[0], n_z) * sigma).to(device)


          z_real = encoder(images_tar).to(device)

          mmd_loss = imq_kernel(z_real.cuda(), z_fake.cuda(), h_dim=2)
          mmd_loss = mmd_loss / batch_size

          total_loss = recon_loss + 0.2 * mmd_loss
          
          total_loss.backward()
          enc_optim.step()
          dec_optim.step()
          
          rec_loss.append(recon_loss.data.item())
          wae_mmd_loss.append(mmd_loss.data.item())
          tot_loss.append(total_loss.data.item())
        if (epoch + 1) % 100 == 0:
          print("Epoch: [%d/%d], Reconstruction Loss: %.4f, MMD Loss %.4f, TOTAL Loss %.4f" %
                  (epoch + 1, ITERS, recon_loss.data.item(),
                  mmd_loss.item(), total_loss.item()))
          
        if epoch+1 == 1:
          os.makedirs('points'+str(n_p)+'_per'+str(n_per))
        if (epoch+1) == 1 or epoch+1 == 5000 or epoch+1==ITERS:
          #plotting
          x1 = []
          x2= []
          x3 = []
          

          dataset_test = data_all[rn_test1]
          for i in range(len(dataset_test)):
            x1.append(dataset_test[i][0])
            x2.append(dataset_test[i][1])
            x3.append(dataset_test[i][2])
            

          
          dataset_test = data_inp[rn_test1]
          

          x1_inp = []
          x2_inp = []
          x3_inp = []
          
          for i in range(len(dataset_test)):
            x1_inp.append(dataset_test[i][0])
            x2_inp.append(dataset_test[i][1])
            x3_inp.append(dataset_test[i][2])
            

          
          data_test = dataset_test
          
          encoder.eval()
          decoder.eval()
          data_test = torch.tensor(data_test)
          images = autograd.Variable(torch.Tensor(data_test)).to(device)
          z = encoder(images)
          x_recon = decoder(z)
          x_recon = x_recon.cpu().detach().numpy()
          print(x_recon.shape)
          x1_rec = []
          x2_rec= []
          x3_rec = []
          for i in range(len(x_recon)):
            x1_rec.append(x_recon[i][0])
            x2_rec.append(x_recon[i][1])
            x3_rec.append(x_recon[i][2])
            
          fig = plt.figure(figsize=(10,10))
          ax = fig.add_subplot(111, projection='3d')
          for i in range(t_samples):
            ax.scatter(x1_rec[i],x2_rec[i],x3_rec[i],c='b', edgecolors = 'black')

          plt.title("reconstructed data")
          plt.savefig('points'+str(n_p)+'_per'+str(n_per)+'/rec_'+str(samples)+'_e'+str(epoch+1)+'.png')
          

          fig = plt.figure(figsize=(10,10))
          ax = fig.add_subplot(111, projection='3d')
          for i in range(t_samples):
            ax.scatter(x1[i],x2[i],x3[i], c='r', edgecolors = 'black')
          plt.title("actual_data")
          plt.savefig('points'+str(n_p)+'_per'+str(n_per)+'/act.png')
          

          fig = plt.figure(figsize=(10,10))
          ax = fig.add_subplot(111, projection='3d')
          for i in range(t_samples):
            ax.scatter(x1_inp[i],x2_inp[i],x3_inp[i], c='g', edgecolors = 'black')
          plt.title("input data")
          
          plt.savefig('points'+str(n_p)+'_per'+str(n_per)+'/inp.png')
          encoder.train()
          decoder.train()

      torch.save(torch.tensor(rec_loss),'./loss/rec_loss/rec_'+str(n_p)+'_e'+str(n_per)+'t'+str(t_samples)+'.pth')
      torch.save(torch.tensor(wae_mmd_loss),'./loss/mmd_loss/mmd_'+str(n_p)+'_e'+str(n_per)+'t'+str(t_samples)+'.pth')
      torch.save(torch.tensor(tot_loss),'./loss/total_loss/tot_'+str(n_p)+'_e'+str(n_per)+'t'+str(t_samples)+'.pth')
      torch.save(encoder.state_dict(),'./loss/encoder/enc_'+str(n_p)+'_e'+str(n_per)+'t'+str(t_samples)+'.pth')
      torch.save(decoder.state_dict(),'./loss/decoder/dec_'+str(n_p)+'_e'+str(n_per)+'t'+str(t_samples)+'.pth')

        
