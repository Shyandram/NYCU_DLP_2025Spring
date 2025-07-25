import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
# from torchvision.utils import save_image
import random
import torch.optim as optim
# from torch import stack

from tqdm import tqdm
# import imageio

import matplotlib.pyplot as plt
from math import log10

from tensorboardX import SummaryWriter

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        # TODO
        # raise NotImplementedError
        self.current_epoch = current_epoch
        self.total_epoch = args.num_epoch
        self.kl_anneal_type = args.kl_anneal_type
        self.kl_anneal_cycle = args.kl_anneal_cycle
        self.kl_anneal_ratio = args.kl_anneal_ratio

        assert self.kl_anneal_type in ["Cyclical", "Monotonic", "WithoutKL"], f"Invalid kl_anneal_type: {self.kl_anneal_type}"
        if self.kl_anneal_type == "Cyclical":
            self.beta = self.frange_cycle_linear(self.total_epoch, start=0.0, stop=self.kl_anneal_ratio, n_cycle=self.kl_anneal_cycle)
        elif self.kl_anneal_type == 'Monotonic':
            self.beta = self.frange_cycle_linear(self.total_epoch, start=0.0, stop=self.kl_anneal_ratio, n_cycle=1)
        elif self.kl_anneal_type == "WithoutKL":
            self.beta = self.frange_cycle_linear(self.total_epoch, start=0.0, stop=self.kl_anneal_ratio, n_cycle=1, without_kl=True) 
        
    def update(self):
        # TODO
        # raise NotImplementedError
        self.current_epoch += 1
    
    def get_beta(self):
        # # TODO
        # raise NotImplementedError
        return self.beta[self.current_epoch]

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1, without_kl=False):
        # TODO
        # raise NotImplementedError
        # https://github.com/haofuml/cyclical_annealing
        L = np.ones(n_iter+1) * stop

        if without_kl:
            return L
        
        period = n_iter/n_cycle
        step = (stop-start)/(period*ratio) # linear schedule

        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i+c*period) < n_iter):
                L[int(i+c*period)] = v
                v += step
                i += 1
        return L 

        

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        self.writer = SummaryWriter(args.save_root)
        
    def forward(self, img, label):
        pass
    
    def training_stage(self):
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            writer_info = {'loss':0, 'beta':0, 'lr':0,}
            
            for (img, label) in (pbar := tqdm(train_loader)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])

                writer_info['loss'] += loss.detach().cpu()

            writer_info['loss'] /= len(train_loader)
            writer_info['beta'] = beta
            writer_info['lr'] = self.scheduler.get_last_lr()[0]
            self.writer.add_scalar('train/loss', writer_info['loss'], self.current_epoch)
            self.writer.add_scalar('train/beta', writer_info['beta'], self.current_epoch)
            self.writer.add_scalar('train/lr', writer_info['lr'], self.current_epoch)
            self.writer.add_scalar('train/tfr', self.tfr, self.current_epoch)

            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
                
            self.eval()
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        writer_info = {'mse':0, 'psnr':0, 'kl':0}
        for (img, label) in (pbar := tqdm(val_loader)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            mse, psnr, kl = self.val_one_step(img, label)
            self.tqdm_bar('val', pbar, mse.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            writer_info['mse'] += mse.detach().cpu()
            writer_info['psnr'] += psnr.detach().cpu()
            writer_info['kl'] += kl.detach().cpu()
            
        writer_info['mse'] /= len(val_loader)
        writer_info['psnr'] /= len(val_loader)
        writer_info['kl'] /= len(val_loader)
        self.writer.add_scalar('val/mse', writer_info['mse'], self.current_epoch)
        self.writer.add_scalar('val/psnr', writer_info['psnr'], self.current_epoch)
        self.writer.add_scalar('val/kl', writer_info['kl'], self.current_epoch)
    
    def training_one_step(self, img, label, adapt_TeacherForcing):
        # TODO
        # raise NotImplementedError

        total_kl_loss = 0
        total_mse_loss = 0
        
        x_hat = img[:, 0]
        for i in range(1, self.args.train_vi_len):
            if adapt_TeacherForcing:
                img_i = img[:, i-1]
            else:
                img_i = x_hat.detach()
            label_i = label[:, i]
            img_current = img[:, i]

            frame_feature = self.frame_transformation(img_i)
            current_frame_feature = self.frame_transformation(img_current)
            label_feature = self.label_transformation(label_i)
            
            z, mu, logvar = self.Gaussian_Predictor(current_frame_feature, label_feature)
            feature = self.Decoder_Fusion(frame_feature, label_feature, z)
            x_hat = self.Generator(feature)

            kl_loss = kl_criterion(mu, logvar, self.batch_size).to(mu.device)

            mse_loss = self.mse_criterion(x_hat, img_current)
            total_mse_loss += mse_loss
            total_kl_loss += kl_loss
        
        self.optim.zero_grad()           
        loss = total_mse_loss + total_kl_loss * self.kl_annealing.get_beta()
        loss.backward()
        self.optimizer_step()
        return loss

    def val_one_step(self, img, label):
        # TODO
        # raise NotImplementedError
            
        total_kl_loss = 0
        total_mse_loss = 0
        total_psnr = 0
        psnrs = []
        
        x_hat = img[:, 0]
        for i in range(1, self.args.val_vi_len):
            img_i = x_hat.detach()
            label_i = label[:, i]
            img_current = img[:, i]            

            frame_feature = self.frame_transformation(img_i)
            current_frame_feature = self.frame_transformation(img_current)
            label_feature = self.label_transformation(label_i)
            
            z, mu, logvar = self.Gaussian_Predictor(current_frame_feature, label_feature)

            feature = self.Decoder_Fusion(frame_feature, label_feature, z)
            x_hat = self.Generator(feature)

            kl_loss = kl_criterion(mu, logvar, self.batch_size).to(mu.device)

            mse_loss = self.mse_criterion(torch.clamp(x_hat, min=0., max=1.), img_current)

            psnr = Generate_PSNR(torch.clamp(x_hat, min=0., max=1.), img_current)

            psnrs.append(psnr.cpu().numpy())
            total_psnr += psnr
            total_mse_loss += mse_loss
            
            total_kl_loss += kl_loss
        total_mse_loss /= (self.args.val_vi_len - 1)
        total_kl_loss /= (self.args.val_vi_len - 1)
        total_psnr /= (self.args.val_vi_len - 1)

        plt.plot(psnrs, label="PSNR per frame")
        plt.xlabel("Frame Index")
        plt.ylabel("PSNR")
        plt.title("PSNR per Frame during Validation")
        plt.legend()
        plt.savefig(os.path.join(self.args.save_root, f"psnr_epoch_latest.png"))
        plt.close()
        
        return total_mse_loss, total_psnr, total_kl_loss
        
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        # TODO
        # raise NotImplementedError
        if self.current_epoch >= self.tfr_sde:
            self.tfr -= self.tfr_d_step
            if self.tfr < 0.0:
                self.tfr = 0.0
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()



def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=4)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=2)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
    

    

    args = parser.parse_args()
    
    main(args)
