import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
import plotly.graph_objects as go
import wandb
from dataset import swiss_role  # Assuming you have a dataset module

# Initialize wandb
wandb.init(project="projectname", entity="username")

def train_model(data_pts, n_samples, ITERS=5000, size_of_batch=2000, latent_dim=2, lr=0.0001, sigma=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rec_loss = []
    wae_mmd_loss = []
    tot_loss = []
    criterion = nn.MSELoss()

    for samples in n_samples:
        encoder, decoder = Encoder(), Decoder()
        encoder.to(device)
        decoder.to(device)

        enc_optim = optim.Adam(encoder.parameters(), lr=lr)
        dec_optim = optim.Adam(decoder.parameters(), lr=lr)

        enc_scheduler = StepLR(enc_optim, step_size=30, gamma=0.5)
        dec_scheduler = StepLR(dec_optim, step_size=30, gamma=0.5)

        # Log the model architecture to wandb
        wandb.watch(encoder, log='all')
        wandb.watch(decoder, log='all')

        # Log hyperparameters to wandb
        wandb.config.update({
            "ITERS": ITERS,
            "size_of_batch": size_of_batch,
            "latent_dim": latent_dim,
            "lr": lr,
            "sigma": sigma
        })

        # For plotting
        t_samples = samples
        t_samples = int(t_samples)
        rn_test1 = np.random.choice(int(samples), int(t_samples))
        data_inp, data_target, data_all = load_data(data_pts, samples)

        for epoch in range(ITERS):
            rec_loss_epoch, wae_mmd_loss_epoch, tot_loss_epoch, z_real, z_fake = train_loop(encoder, decoder, enc_optim, dec_optim, criterion, data_inp, data_target, size_of_batch, latent_dim, device, sigma)

            rec_loss.append(rec_loss_epoch)
            wae_mmd_loss.append(wae_mmd_loss_epoch)
            tot_loss.append(tot_loss_epoch)

            # Log losses to wandb
            wandb.log({
                "Reconstruction Loss": rec_loss_epoch,
                "WAE MMD Loss": wae_mmd_loss_epoch,
                "Total Loss": tot_loss_epoch,
                "Epoch": epoch + 1
            })

            if (epoch + 1) % 100 == 0:
                print("Epoch: [%d/%d], Reconstruction Loss: %.4f, MMD Loss %.4f, TOTAL Loss %.4f" %
                      (epoch + 1, ITERS, rec_loss_epoch, wae_mmd_loss_epoch, tot_loss_epoch))

            if epoch+1 == 1:
                directory_path = "points"
                latent_folder = "latent_space"
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                    print(f"Directory '{directory_path}' was created.")
                else:
                    print(f"Directory '{directory_path}' already exists.")

                if not os.path.exists(latent_folder):
                    os.makedirs(latent_folder)
                    print(f"Directory '{latent_folder}' was created.")
                else:
                    print(f"Directory '{latent_folder}' already exists.")

            if (epoch+1) == 1 or ((epoch+1) % 500) == 0 or epoch+1 == ITERS:
                # Plotting
                x1 = []
                x2 = []
                x3 = []

                # Visualization
                dataset_test = data_all[rn_test1]
                data_test = dataset_test
                for i in range(len(dataset_test)):
                    x1.append(dataset_test[i][0])
                    x2.append(dataset_test[i][1])
                    x3.append(dataset_test[i][2])

                data_test = torch.tensor(data_test)
                images = autograd.Variable(torch.Tensor(data_test)).to(device)
                z = encoder(images)
                x_recon = decoder(z)
                x_recon = x_recon.cpu().detach().numpy()
                print(x_recon.shape)
                x1_rec = []
                x2_rec = []
                x3_rec = []
                for i in range(len(x_recon)):
                    x1_rec.append(x_recon[i][0])
                    x2_rec.append(x_recon[i][1])
                    x3_rec.append(x_recon[i][2])

                # Example usage during your training loop after reconstruction:
                actual_data = np.array([x1, x2, x3]).T
                reconstructed_data = np.array([x1_rec, x2_rec, x3_rec]).T
                plot_latent_space_qq(z_real, z_fake, samples, epoch, latent_folder)
                plot_latent_space_scatter(z_real, samples, epoch, latent_folder)
                plot_actual_vs_reconstructed_3d(x1, x2, x3, x1_rec, x2_rec, x3_rec, epoch, samples, directory_path)
                plot_actual_vs_reconstructed_sidebyside(actual_data, reconstructed_data, epoch, samples, directory_path)
                plot_histogram(z_real, z_fake, samples, epoch, latent_folder)

    # Save the losses after training
    torch.save(torch.tensor(rec_loss), 'rec_loss.pt')
    torch.save(torch.tensor(wae_mmd_loss), 'wae_mmd_loss.pt')
    torch.save(torch.tensor(tot_loss), 'tot_loss.pt')

def plot_losses(rec_loss, wae_mmd_loss, tot_loss):
    # Plot Reconstruction Loss
    plt.figure()
    plt.plot(rec_loss, label='Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Reconstruction Loss')
    plt.savefig('reconstruction_loss.png')
    plt.close()
    
    # Plot WAE MMD Loss
    plt.figure()
    plt.plot(wae_mmd_loss, label='WAE MMD Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('WAE MMD Loss')
    plt.savefig('wae_mmd_loss.png')
    plt.close()
    
    # Plot Total Loss
    plt.figure()
    plt.plot(tot_loss, label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Total Loss')
    plt.savefig('total_loss.png')
    plt.close()

def train_loop(encoder, decoder, enc_optim, dec_optim, criterion, data_inp, data_target, size_of_batch, latent_dim, device, sigma):
    rec_loss_epoch = 0
    wae_mmd_loss_epoch = 0
    tot_loss_epoch = 0

    ub_list = np.arange(0, data_inp.shape[0] + size_of_batch, size_of_batch)
    
    for steps in range(data_inp.shape[0] // size_of_batch):
        _data = data_inp[ub_list[steps]:ub_list[steps+1]][:]
        _data_tar = data_target[ub_list[steps]:ub_list[steps+1]][:]
        
        images = autograd.Variable(torch.Tensor(_data)).to(device)
        images_tar = autograd.Variable(torch.Tensor(_data_tar)).to(device)

        enc_optim.zero_grad()
        dec_optim.zero_grad()

        batch_size = images.size()[0]
        z = encoder(images)
        x_recon = decoder(z)

        recon_loss = criterion(x_recon, images_tar)
        z_fake = Variable(torch.rand(images.size()[0], latent_dim) * sigma).to(device)
        z_real = encoder(images_tar).to(device)
        mmd_loss = imq_kernel(z_real, z_fake, h_dim=2) / batch_size

        total_loss = recon_loss + 0.2 * mmd_loss
        total_loss.backward()
        enc_optim.step()
        dec_optim.step()

        rec_loss_epoch += recon_loss.item()
        wae_mmd_loss_epoch += mmd_loss.item()
        tot_loss_epoch += total_loss.item()

    return rec_loss_epoch, wae_mmd_loss_epoch, tot_loss_epoch, z_real, z_fake

def load_data(data_pts, samples, data_pts_load = True):
    # Load or generate data here
    print("data_pts:",data_pts)

    if data_pts_load == False:
        data_all = swiss_role(data_pts)
        torch.save(torch.tensor(data_all),'./datapts_sample.pth')
    elif data_pts_load == True:
        data_all = torch.load('./datapts_sample.pth').numpy()
    print("actual data length:", len(data_all))

    rn = np.random.choice(len(data_all), samples)
    data_inp = data_all[rn]
    data_target = data_all[rn]

    return data_inp, data_target, data_all

def plot_actual_vs_reconstructed_3d(x1,x2,x3, x1_rec, x2_rec, x3_rec, epoch, samples, folder_path):
    fig = plt.figure(figsize=(12, 6))
    directory_path = os.path.join(folder_path, "3d") 
    if not os.path.exists(directory_path):
      os.makedirs(directory_path)
      print(f"Directory '{directory_path}' was created.")
    else:
      print(f"Directory '{directory_path}' already exists.")
    # Plot for reconstructed data
    scatter = go.Scatter3d(x=x1_rec, y=x2_rec, z=x3_rec, mode='markers', marker=dict(color='blue', size=5))

    # Create the layout for the plot
    layout = go.Layout(scene=dict(aspectmode="cube"))

    # Create the figure and add the trace
    fig = go.Figure(data=[scatter], layout=layout)
    fig.write_html(os.path.join(directory_path, f'rec_'+str(samples)+'_e'+str(epoch+1)+'.html'))

    # Plot for actual data
    #plotly----------------
    if epoch + 1 == 1:
      scatter = go.Scatter3d(x=x1, y=x2, z=x3, mode='markers', marker=dict(color='red', size=5))

      # Create the layout for the plot
      layout = go.Layout(scene=dict(aspectmode="cube"))

      # Create the figure and add the trace
      fig = go.Figure(data=[scatter], layout=layout)

      fig.write_html(os.path.join(directory_path, 'act.html'))

def plot_actual_vs_reconstructed_sidebyside(actual_data, reconstructed_data, epoch, samples, folder_path):
    fig = plt.figure(figsize=(12, 6))
    directory_path = os.path.join(folder_path, "2d") 
    if not os.path.exists(directory_path):
      os.makedirs(directory_path)
      print(f"Directory '{directory_path}' was created.")
    else:
      print(f"Directory '{directory_path}' already exists.")
    
    # Plot for actual data
    # if epoch + 1 == 1:
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(actual_data[:, 0], actual_data[:, 1], actual_data[:, 2], c='r', label='Actual', edgecolors='black')
    ax1.set_title('Actual Data')
    ax1.legend()

    # Plot for reconstructed data
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(reconstructed_data[:, 0], reconstructed_data[:, 1], reconstructed_data[:, 2], c='b', label='Reconstructed', edgecolors='black')
    ax2.set_title('Reconstructed Data')
    ax2.legend()

    # General settings
    plt.suptitle(f'Epoch: {epoch+1}, Samples: {samples}')
    plt.savefig(os.path.join(directory_path, f'compare_{samples}_epoch_{epoch+1}.png'))

def plot_histogram(z_real, z_fake, samples, epoch, latent_folder):

    directory_path = os.path.join(latent_folder, "histogram")
    if not os.path.exists(directory_path):
      os.makedirs(directory_path)
      print(f"Directory '{directory_path}' was created.")
    else:
      print(f"Directory '{directory_path}' already exists.")

    gauss_samples = np.array(z_fake.cpu().detach().numpy())
    min_value = torch.min(z_real)
    max_value = torch.max(z_real)
    norm_z_real = (z_real - min_value) / (max_value - min_value)
    enc_samples = np.array(norm_z_real.cpu().detach().numpy())

    # Extract the values for each dimension separately
    x_gauss = gauss_samples[:, 0]
    y_gauss = gauss_samples[:, 1]
    x_enc = enc_samples[:, 0]
    y_enc = enc_samples[:, 1]
    
    # Define the number of bins for the histograms
    num_bins = 20

    # Create subplots for the central and marginal distributions
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)

    # Plot the central distribution (mixture of both)
    ax_main = fig.add_subplot(gs[1:3, 0:2])
    ax_main.hist2d(x_gauss, y_gauss, bins=num_bins, cmap='Blues')
    ax_main.hist2d(x_enc, y_enc, bins=num_bins, cmap='Oranges')
    ax_main.set_xlabel('x')
    ax_main.set_ylabel('y')
    ax_main.set_title('Mixture of Marginal Distributions')

    # Plot the histogram for the first dimension (x) on the top
    ax_top = fig.add_subplot(gs[0, 0:2])
    ax_top.hist(x_gauss, bins=num_bins, alpha=0.5, color='blue')
    ax_top.hist(x_enc, bins=num_bins, alpha=0.5, color='yellow')
    ax_top.set_xlabel('x')
    ax_top.set_ylabel('Frequency')
    ax_top.set_title('Marginal Distribution: Dimension 1 (x)')

    # Plot the histogram for the second dimension (y) on the right
    ax_right = fig.add_subplot(gs[1:3, 2])
    ax_right.hist(y_gauss, bins=num_bins, orientation='horizontal', alpha=0.5, color='blue')
    ax_right.hist(y_enc, bins=num_bins, orientation='horizontal', alpha=0.5, color='yellow')
    ax_right.set_xlabel('Frequency')
    ax_right.set_ylabel('y')
    ax_right.set_title('Marginal Distribution: Dimension 2 (y)')

    # Remove unnecessary spines and ticks
    ax_main.spines['right'].set_visible(False)
    ax_main.spines['top'].set_visible(False)
    ax_main.tick_params(right=False, top=False)

    ax_top.spines['bottom'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.tick_params(bottom=False, right=False)

    ax_right.spines['top'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    ax_right.tick_params(top=False, left=False)

    fig.savefig(os.path.join(directory_path, "overlap_"+str(epoch)+".png"))

def plot_latent_space_qq(z_real, z_fake, samples, epoch, latent_folder):
    directory_path = os.path.join(latent_folder, "qq")
    
    if not os.path.exists(directory_path):
      os.makedirs(directory_path)
      print(f"Directory '{directory_path}' was created.")
    else:
      print(f"Directory '{directory_path}' already exists.")

    min_value = torch.min(z_real)
    max_value = torch.max(z_real)
    norm_z_real = (z_real - min_value) / (max_value - min_value)
    z_real_arr = norm_z_real.cpu().detach().numpy()
    x_z_real = z_real_arr[:,0]
    y_z_real = z_real_arr[:,1]
    z_real_quant = np.linspace(0,1, len(z_real_arr))
    p_quant = np.linspace(0,1, len(z_fake))
    sorted_z_real = np.sort(z_real_arr, axis = 0)
    p_q_val = np.array([stats.mstats.mquantiles(sorted_z_real[:,i], prob = p_quant) for i in range(2)])

    fig = plt.figure(figsize = (10,10))
    plt.scatter(p_q_val[0], p_q_val[1], color='blue', label = 'p')
    plt.plot(z_real_quant, z_real_quant, color='red', label = 'z')
    plt.xlabel('Quantiles of z_real')
    plt.ylabel('Quantiles of P')
    plt.title('Bivariate QQ Plot')
    plt.legend()
    fig.savefig(os.path.join(directory_path, f"overlap_"+str(epoch)+"_"+str(samples)+".png"))

def plot_latent_space_scatter(z_real, samples, epoch, latent_folder):
    directory_path = os.path.join(latent_folder, "scatter")
    
    if not os.path.exists(directory_path):
      os.makedirs(directory_path)
      print(f"Directory '{directory_path}' was created.")
    else:
      print(f"Directory '{directory_path}' already exists.")

    z_real_np = z_real.cpu().detach().numpy()
    fig, ax = plt.subplots(figsize=(8, 8))
    
    scatter = ax.scatter(z_real_np[:, 0], z_real_np[:, 1], c=np.linspace(0, 1, len(z_real_np)), cmap='viridis')
    ax.set_title(f'Latent Space Representation - Epoch {epoch+1}')
    ax.set_xlabel('Quantiles of z_real')
    ax.set_ylabel('Quantiles of P')
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label('Index Gradient')
    
    plt.savefig(os.path.join(directory_path, f'latent_space_epoch_{epoch+1}_{samples}.png'))

class Encoder(nn.Module):
  def __init__(self):
    super().__init__()

    # seven hidden layers
    self.enc_1 = torch.nn.Linear(3, 6)
    self.enc_2 = torch.nn.Linear(6, 12)
    self.enc_3 = torch.nn.Linear(12, 32)
    self.enc_4 = torch.nn.Linear(32, 64)
    self.enc_5 = torch.nn.Linear(64, 32)
    self.enc_6 = torch.nn.Linear(32, 12)
    self.enc_7 = torch.nn.Linear(12, 2)
    #self.enc_4 = torch.nn.Linear(125, 64)
    #self.enc_5 = torch.nn.Linear(64, 32)

  def forward(self, x):
    x = F.leaky_relu(self.enc_1(x))
    x = F.leaky_relu(self.enc_2(x))
    x = F.leaky_relu(self.enc_3(x))
    x = F.leaky_relu(self.enc_4(x))
    x = F.leaky_relu(self.enc_5(x))
    x = F.leaky_relu(self.enc_6(x))
    x = F.leaky_relu(self.enc_7(x))
    #x = F.leaky_relu(self.enc_4(x))
    #x = F.leaky_relu(self.enc_5(x))
    return x

class Decoder(nn.Module):
  def __init__(self):
    super().__init__()

    #four layers
    self.dec_1 = torch.nn.Linear(2, 32)
    self.dec_2 = torch.nn.Linear(32, 64)
    self.dec_3 = torch.nn.Linear(64, 32)
    self.dec_4 = torch.nn.Linear(32, 3)
    #self.dec_4 = torch.nn.Linear(250, 500)
    #self.dec_5 = torch.nn.Linear(500, 512)

  def forward(self, x):
    x = F.leaky_relu(self.dec_1(x))
    x = F.leaky_relu(self.dec_2(x))
    x = F.leaky_relu(self.dec_3(x))
    x = self.dec_4(x)
    #x = F.leaky_relu(self.dec_4(x))
    #x = F.leaky_relu(self.dec_5(x))
    # x = F.leaky_relu(x)
    return x

def imq_kernel(X, Y, h_dim):
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

def main():
    data_pts = 50000
    n_samples = [30000]  # You can adjust this list as needed
    ITERS = 10000
    size_of_batch = 5000
    latent_dim = 2
    lr = 0.0001
    sigma = 1
    
    train_model(data_pts, n_samples, ITERS, size_of_batch, latent_dim, lr, sigma)

if __name__ == "__main__":
    main()
