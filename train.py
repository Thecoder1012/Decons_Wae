import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import torch.autograd as autograd
import matplotlib.pyplot as plt
import random
import numpy as np
import time
from model import Encoder, Decoder, Encoder_Mnist, Decoder_Mnist
from dataset import inf_train_gen, mnist_loader
from loss import imq_kernel, rbf_kernel, jenson_shannon_divergence
import config
import argparse
import os
import sys

def train_g(gps, js, exp, beta, gauss, mnist):

    #-------initialising device--------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    if mnist == 0:
        #------data loading--------
        if config.data_pts_load == False:
            data_all = inf_train_gen(data_pts)
            torch.save(torch.tensor(data_all),'datapts.pth')
            print('data generated')

        elif config.data_pts_load == True:
            data_all = torch.load('datapts.pth').numpy()
            print('data loaded')


        for i in config.n_samples:
            for model_num in range(config.total_epoch):
                rec_loss = []
                wae_cust_loss = []
                tot_loss = []
                rn = np.random.choice(config.total_corpus, i)
                data = data_all[rn]
                print(data.shape)
                encoder, decoder = Encoder(), Decoder()
                pytorch_total_params = sum(p.numel() for p in encoder.parameters())
                print(pytorch_total_params)
                pytorch_total_params = sum(p.numel() for p in decoder.parameters())
                print(pytorch_total_params)
                criterion = nn.MSELoss()

                criterion.to(device)
                encoder.to(device)
                decoder.to(device)

                encoder.train()
                decoder.train()
                enc_optim = optim.Adam(encoder.parameters(), lr= config.lr)
                dec_optim = optim.Adam(decoder.parameters(), lr= config.lr)

                enc_scheduler = StepLR(enc_optim, step_size=30, gamma=0.5)
                dec_scheduler = StepLR(dec_optim, step_size=30, gamma=0.5)

                if torch.cuda.is_available():
                    encoder, decoder = encoder.cuda(), decoder.cuda()
                    data = torch.tensor(data)
                    ub_list = []
                    ub_list = np.arange(0, data.shape[0] + config.size_of_batch_g, config.size_of_batch_g)
                
                for epoch in range(config.ITERS_g):
                    total_steps = data.shape[0] // config.size_of_batch_g
                    if epoch ==0:
                        start = time.process_time()
                    for steps in range(data.shape[0] // config.size_of_batch_g):
                        _data = data[ub_list[steps]:ub_list[steps+1]][:]
                        images = autograd.Variable(torch.Tensor(_data)).to(device)
                        
                        enc_optim.zero_grad()
                        dec_optim.zero_grad()

                        # ================Recons loss============ #
                        batch_size = images.size()[0]
                        z = encoder(images, gps)
                        x_recon = decoder(z, gps)
                        recon_loss = criterion(x_recon.cuda(), images.cuda())

                        # ======== Kernel Loss ======== #
                        z_fake = Variable(torch.rand(images.size()[0], config.n_z_g) * config.sigma).to(device)
                        z_real = encoder(images).to(device)
                        
                        if js == 0 :
                            cust_loss = imq_kernel(z_real.cuda(), z_fake.cuda(), h_dim=2)
                            cust_loss = cust_loss / batch_size
                            total_loss = recon_loss + config.alpha * cust_loss 

                        elif js == 1:
                            if exp == 1 and beta == 0 and gauss == 0:
                                p = Variable(torch.exp(torch.rand(images.size()[0], config.n_z_g)) * config.sigma).to(device)
                            elif exp ==0 and beta == 1 and gauss == 0:
                                beta_dis = torch.distributions.Beta(2,2).sample(torch.Size([images.size()[0], config.n_z_g])).cuda()
                                p = Variable(beta_dis * config.sigma).cuda()
                            elif exp == 0 and beta == 0 and gauss == 1:
                                p = Variable(torch.rand(images.size()[0], config.n_z_g) * config.sigma).to(device)
                            else:
                                print("please change your selections")
                                exit()

                            q = encoder(images).to(device)
                            cust_loss = jenson_shannon_divergence(p,q)
                            total_loss = recon_loss + config.alpha * cust_loss

                        total_loss.backward()
                        enc_optim.step()
                        dec_optim.step()
                        
                        rec_loss.append(recon_loss.data.item())
                        wae_cust_loss.append(cust_loss.data.item())
                        tot_loss.append(total_loss.data.item())
                    
                    if epoch == 0:
                        print("epoch "+ str(epoch) +" ",time.process_time() - start)

                    if (epoch + 1) % config.recons_ep == 0:
                        print("Model Number: [%d/%d] Epoch: [%d/%d], Reconstruction Loss: %.4f, MMD Loss %.4f, TOTAL Loss %.4f" %
                                (model_num + 1, config.total_epoch, epoch + 1, config.ITERS_g, recon_loss.data.item(),
                                cust_loss.item(), total_loss.item()))
                            
                    if epoch == config.ITERS_g -1:
                        path = str(i)
                        torch.save(torch.tensor(rec_loss),path+'/rec_loss/rec_'+str(model_num)+'_'+str(i)+'.pth')
                        torch.save(torch.tensor(wae_cust_loss),path+'/cust_loss/cust_'+str(model_num)+'_'+str(i)+'.pth')
                        torch.save(torch.tensor(tot_loss),path+'/total_loss/total_'+str(model_num)+'_'+str(i)+'.pth')
                        torch.save(encoder.state_dict(),path+'/encoder/enc_'+str(model_num)+'_'+str(i)+'.pth')
                        torch.save(decoder.state_dict(),path+'/decoder/dec_'+str(model_num)+'_'+str(i)+'.pth')

    elif mnist == 1:
        train_m(gps, js, exp, beta, gauss, mnist)

def train_m(gps, js, exp, beta, gauss, mnist):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    if mnist == 1:
        #Converting data to torch.FloatTensor
        train_data, test_data, classes = mnist_loader()
        encoder, decoder = Encoder_Mnist().to(device), Decoder_Mnist().to(device)
        criterion = nn.MSELoss()

        criterion.to(device)

        encoder.to(device)
        decoder.to(device)

        encoder.train()
        decoder.train()

        enc_optim = optim.Adam(encoder.parameters(), lr= config.lr)
        dec_optim = optim.Adam(decoder.parameters(), lr= config.lr)

        enc_scheduler = StepLR(enc_optim, step_size=30, gamma=0.5)
        dec_scheduler = StepLR(dec_optim, step_size=30, gamma=0.5)
        path = "test_v2_bs_"+str(config.size_of_batch_m)+"ls_"+str(config.n_z_m)+"ac_relu/"
        os.mkdir(path)
        original_stdout = sys.stdout

        for sample_size in config.n_samples:
            rec_loss_list = []
            cust_loss_list = []
            total_loss_list = []
            total_models = 20
            for model_num in range(total_models):
                cnt = -1
                subset_indices = random.sample(range(0, len(train_data)), sample_size)
                subset = torch.utils.data.Subset(train_data, subset_indices)
                train_loader_subset = torch.utils.data.DataLoader(subset, batch_size=1000, num_workers=0, shuffle=False)
                for epoch in range(config.ITERS_m):
                    for data in train_loader_subset:
                        images, _ = data
                        images = images.to(device)
                        enc_optim.zero_grad()
                        dec_optim.zero_grad()
                        x_latent = encoder(images, gps = 1).to(device)
                        x_recon = decoder(x_latent, gps = 1).to(device)

                        recon_loss = criterion(x_recon, images).to(device)
                        
                        if js == 0 :
                            x_fake = Variable(torch.rand(images.size()[0], config.n_z_m) * config.sigma).to(device)
                            cust_loss = imq_kernel(x_latent.cuda(), x_fake.cuda(), h_dim=2)
                            cust_loss = cust_loss / batch_size
                            total_loss = recon_loss + config.alpha * cust_loss 

                        if js == 1:
                            if exp == 1 and beta == 0 and gauss == 0:
                                p = Variable(torch.exp(torch.rand(images.size()[0], config.n_z_m)) * config.sigma).to(device)
                            elif exp ==0 and beta == 1 and gauss == 0:
                                beta_dis = torch.distributions.Beta(2,2).sample(torch.Size([images.size()[0], config.n_z_m])).cuda()
                                p = Variable(beta_dis * config.sigma).cuda()
                            elif exp == 0 and beta == 0 and gauss == 1:
                                p = Variable(torch.rand(images.size()[0], config.n_z_m) * config.sigma).to(device)
                            else:
                                print("please change your selections")
                                exit()
                                
                            cust_loss = jenson_shannon_divergence(p,x_latent).to(device)
                            total_loss = recon_loss + config.alpha * cust_loss
                            total_loss.backward()

                        enc_optim.step()
                        dec_optim.step()

                    if epoch+1 == 1 and model_num + 1 ==1:
                        os.makedirs(path+'loss_'+str(sample_size))
                        os.makedirs(path+'./loss_'+str(sample_size)+'/rec_loss')
                        os.makedirs(path+'./loss_'+str(sample_size)+'/cust_loss')
                        os.makedirs(path+'./loss_'+str(sample_size)+'/total_loss')
                        os.makedirs(path+'./loss_'+str(sample_size)+'/encoder')
                        os.makedirs(path+'./loss_'+str(sample_size)+'/decoder')
                
                    if (epoch+1) % 60 == 0:
                        print("Model Number [%d/%d], Epoch: [%d/%d], Reconstruction Loss: %.6f JS Loss: %.6f Total Loss: %.6f" %(model_num+1, total_models, epoch + 1, config.ITERS_m, recon_loss.data.item(), cust_loss.item(), total_loss))
                
                    if epoch == config.ITERS_m - 1:
                        torch.save(torch.tensor(rec_loss_list), path+'./loss_'+str(sample_size)+'/rec_loss'+'/rec_'+str(model_num)+'_loss.pth')
                        torch.save(torch.tensor(cust_loss_list), path+'./loss_'+str(sample_size)+'/cust_loss'+'/cust_'+str(model_num)+'_loss.pth')
                        torch.save(torch.tensor(total_loss_list), path+'./loss_'+str(sample_size)+'/total_loss'+'/total_'+str(model_num)+'_loss.pth')
                        torch.save(encoder.state_dict(),path+'./loss_'+str(sample_size)+'/encoder/enc_'+str(model_num)+'.pth')
                        torch.save(decoder.state_dict(),path+'./loss_'+str(sample_size)+'/decoder/dec_'+str(model_num)+'.pth') 
                        print(str(sample_size)+' Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--groupsort", type=int, default = 0)
    parser.add_argument("--js", type=int, default = 0)
    parser.add_argument("--beta", type=int, default = 0)
    parser.add_argument("--exp", type=int, default = 0)
    parser.add_argument("--gauss", type=int, default = 0)
    parser.add_argument("--mnist", type=int, default = 0)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parsed argument
    c_gps = args.groupsort
    c_js = args.js
    c_beta = args.beta
    c_gauss = args.gauss
    c_exp = args.exp
    c_mnist = args.mnist
    
    if c_mnist == 0:
        train_g(c_gps, c_js, c_exp, c_beta, c_gauss, c_mnist)
    else:
        train_m(c_gps, c_js, c_exp, c_beta, c_gauss, c_mnist)