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
from dataset import swiss_role, add_noise
import numpy as np

# Set the environment variable for CuBLAS
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # or ':16:8'

random.seed(800)
np.random.seed(800)
torch.manual_seed(800)
torch.use_deterministic_algorithms(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class Encoder(nn.Module):

class Decoder(nn.Module):

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

data_pts = 10000
# noise_points_list = [2000]
print("data_pts:",data_pts)
data_pts_load = True
if data_pts_load == False:
    data = swiss_role(data_pts)
    torch.save(torch.tensor(data),f'./Swiss_role/datapts_sample_{data_pts}.pth')
elif data_pts_load == True:
    data_all = torch.load('./Swiss_role/datapts_sample.pth').numpy()
    data = data_all[0:data_pts]
print("actual data length:", len(data_all))

#result folder creation
parent_dir = "./points"
os.makedirs(parent_dir)
total_models = 1
n_samples = [1000]
noise_per_list = [0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.9]
for noise_samples in n_samples:
  #n_samples = [50000]
  for noise_per in noise_per_list:
    print("data_pts:",data_pts)
    print('noise points:',noise_samples)
    print('noise percentage:',noise_per)
    noise_points = data[(data_pts - noise_samples): data_pts]
    data_noise = data[0: (data_pts - noise_samples)]
    data_noise = data_noise.tolist()
    data_noise = add_noise(data_noise, noise_samples, noise_points, noise_type = "cauchy", noise = noise_per)
    for num_model in range(total_models):
      print("number of samples:",data_pts)
      print("Training model no:", num_model)
      print("Noise Percentage:", noise_per)
      epoch_choice = []
      size_of_batch = 1000
      rec_loss = []
      wae_mmd_loss = []
      tot_loss = []
      ITERS = 3000  #100000
      n_z = 2
      sigma = 1
      step = 0
      epoch = 0
      data_inp = data_noise
      data_target = data
      encoder, decoder = Encoder(), Decoder()
      pytorch_total_params = sum(p.numel() for p in encoder.parameters())
      #print(pytorch_total_params)
      pytorch_total_params = sum(p.numel() for p in decoder.parameters())
      #print(pytorch_total_params)

      criterion = nn.MSELoss()

      criterion.to(device)
      encoder.to(device)
      decoder.to(device)

      encoder.train()
      decoder.train()
      #applying AdamW
      enc_optim = optim.AdamW(encoder.parameters(), lr= 0.0001)
      dec_optim = optim.AdamW(decoder.parameters(), lr= 0.0001)

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
      rn_test1 = np.random.choice(10000,int(t_samples))
        
      for epoch in range(ITERS):
        #lb = 0
        #ub = size_of_batch
        total_steps = data_inp.shape[0] // size_of_batch
        #print(total_steps)
        for steps in range(data_inp.shape[0] // size_of_batch):
          _data = data_inp[ub_list[steps]:ub_list[steps+1]][:]
          _data_tar = data_target[ub_list[steps]:ub_list[steps+1]][:]
          #lb = ub
          #ub = ub + size_of_batch
          #ub = ub_list[]
          #print(_data.shape, lb, ub)
          #real_data = torch.Tensor(_data)
          images = autograd.Variable(torch.Tensor(_data)).to(device)
          images_tar = autograd.Variable(torch.Tensor(_data_tar)).to(device)

          enc_optim.zero_grad()
          dec_optim.zero_grad()

          # ======== Train Generator ======== #

          batch_size = images.size()[0]
        #   print(batch_size)

          z = encoder(images)

          x_recon = decoder(z)


          recon_loss = criterion(x_recon.cuda(), images_tar.cuda())

          # ======== MMD Kernel Loss ======== #

          z_fake = Variable(torch.rand(images.size()[0], n_z) * sigma).to(device)


          z_real = encoder(images_tar).to(device)
          
          mmd_loss = imq_kernel(z_real.cuda(), z_fake.cuda(), h_dim=2)
          mmd_loss = mmd_loss / batch_size

          total_loss = recon_loss + 0.2 * mmd_loss
          # lambda*mmd_loss + wasserstein loss
          total_loss.backward()
          enc_optim.step()
          dec_optim.step()
          #step += 1
          #print(step)
          rec_loss.append(recon_loss.data.item())
          wae_mmd_loss.append(mmd_loss.data.item())
          tot_loss.append(total_loss.data.item())
        if (epoch + 1) % 100 == 0:
          print("Epoch: [%d/%d], Reconstruction Loss: %.4f, MMD Loss %.4f, TOTAL Loss %.4f" %
                  (epoch + 1, ITERS, recon_loss.data.item(),
                  mmd_loss.item(), total_loss.item()))

        #for plotting  
        if epoch+1 == 1:
          os.makedirs('points'+str(noise_samples)+'_per'+str(noise_per))
        
        if (epoch+1) == 1 or ((epoch+1) % 100) == 0 or epoch+1==ITERS:
          #plotting
          x1 = []
          x2= []
          x3 = []
          

          dataset_test = data[rn_test1]

          #dataset_test = data_all[59400:60000]

          #dataset_actual = data_all[rn_test1]

          #print(dataset_test.shape)
          #print(rn_test1)
          for i in range(len(dataset_test)):
            x1.append(dataset_test[i][0])
            x2.append(dataset_test[i][1])
            x3.append(dataset_test[i][2])
            

          #dataset_test = dataset_noise[rn_test1]
          dataset_test = data_inp[rn_test1]
          #print(dataset_test.shape)

          x1_inp = []
          x2_inp = []
          x3_inp = []
          #print(rn_test1)
          for i in range(len(dataset_test)):
            x1_inp.append(dataset_test[i][0])
            x2_inp.append(dataset_test[i][1])
            x3_inp.append(dataset_test[i][2])
            

          #plt.axis('off')
          #plt.savefig("Test_data.png")
          data_test = dataset_test
          #print(data_test.shape)
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
          plt.savefig('points'+str(noise_samples)+'_per'+str(noise_per)+'/rec_2000_e'+str(epoch+1)+'.png')
          #plt.show()

          fig = plt.figure(figsize=(10,10))
          ax = fig.add_subplot(111, projection='3d')
          for i in range(t_samples):
            ax.scatter(x1[i],x2[i],x3[i], c='r', edgecolors = 'black')
          plt.title("actual_data")
          plt.savefig('points'+str(noise_samples)+'_per'+str(noise_per)+'/act.png')
          #plt.show()

          fig = plt.figure(figsize=(10,10))
          ax = fig.add_subplot(111, projection='3d')
          for i in range(t_samples):
            ax.scatter(x1_inp[i],x2_inp[i],x3_inp[i], c='g', edgecolors = 'black')
          plt.title("input data")
          
          plt.savefig('points'+str(noise_samples)+'_per'+str(noise_per)+'/inp.png')
          encoder.train()
          decoder.train()

