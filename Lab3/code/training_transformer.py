import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tensorboardX import SummaryWriter

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim,self.scheduler = self.configure_optimizers()
        self.prepare_training()
        self.args = args
        self.loss = nn.CrossEntropyLoss()

        self.scalar = GradScaler(enabled=args.amp)
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        loader = tqdm(train_loader)
        for i, x in enumerate(loader):
            x = x.to(args.device)
            self.optim.zero_grad()
            ratio = np.random.rand()
            with autocast(device_type=args.device, enabled=self.args.amp):
                logits, z_indices = self.model(x, ratio)
                loss = self.loss(logits.permute(0, 2, 1), z_indices)
            self.scalar.scale(loss).backward()

            self.scalar.step(self.optim)

            self.scalar.update()

            total_loss += loss.item()

            loader.set_description(f"Training Loss: {total_loss / (i + 1)}")        
        self.scheduler.step()
        return total_loss / (i + 1)

    @torch.no_grad()
    def eval_one_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0
        loader = tqdm(val_loader)
        for i, x in enumerate(loader):
            x = x.to(args.device)
            ratio = np.random.rand()
            logits, z_indices = self.model(x, ratio)
            loss = self.loss(logits.permute(0, 2, 1), z_indices)
            total_loss += loss.item()
            loader.set_description(f"Validation Loss: {total_loss / (i + 1)}")
        return total_loss / (i + 1)




    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        return optimizer,scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./lab3_dataset/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./lab3_dataset/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(default: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Start training from ** epoch(default: 0)')
    parser.add_argument('--ckpt-interval', type=int, default=1, help='Save CKPT per ** epochs(default: 1)')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    parser.add_argument('--amp', action='store_true', help='Enable AMP.', default=True)

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
    #TODO2 step1-5:    
    writer = SummaryWriter()
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        print(f"Epoch {epoch}")
        print("Training")
        training_loss = train_transformer.train_one_epoch(train_loader)
        writer.add_scalar("Loss/Train", training_loss, epoch)
        print("Validation")
        valid_loss = train_transformer.eval_one_epoch(val_loader)
        writer.add_scalar("Loss/Validation", valid_loss, epoch)

        writer.add_scalar("Learning Rate", train_transformer.scheduler.get_last_lr()[0], epoch)

        if epoch % args.save_per_epoch == 0:
            torch.save(train_transformer.model.transformer.state_dict(), f"transformer_checkpoints/ckpt_{epoch}.pt")
        if epoch % args.ckpt_interval == 0:
            torch.save(train_transformer.model.transformer.state_dict(), f"transformer_checkpoints/last_ckpt.pt")