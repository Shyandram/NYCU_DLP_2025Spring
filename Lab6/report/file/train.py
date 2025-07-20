import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.amp import autocast, GradScaler

from diffusers import DDPMScheduler

from ddpm import cDDPM
from dataset import iclevr
from tensorboardX import SummaryWriter

def train(args):
    # Set up the save directory
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
            
    # Set up the TensorBoard writer
    log_dir = os.path.join(save_dir, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up the dataset and dataloader
    train_dataset = iclevr(
        root=args.dataset, 
        image_size=args.image_size,
        mode='train'
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Set up the model
    model = cDDPM(
        sample_size=args.image_size,
        n_channel=args.n_channel,
        depth=args.depth,
    ).to(device)
    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"Loaded model from {args.ckpt}")

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule=args.noise_schedule,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    criterion = nn.MSELoss()

    scaler = GradScaler(enabled=args.amp) if args.amp else None

    # Training loop
    best_loss = float('inf')
    for epoch in range(args.start_epoch, args.num_epochs):    
        model.train()
        total_loss = 0.
        loader = tqdm(train_loader)
        for i, (images, labels) in enumerate(loader):
            optimizer.zero_grad()

            images = images.to(device)
            labels = labels.float().to(device)
            # Forward 
            noise = torch.randn_like(images)
            t = torch.randint(0, args.timesteps, (images.size(0),), device=device).long()
            noisy_images = noise_scheduler.add_noise(images, noise, t)

            with autocast(device_type=device.type, enabled=args.amp):
                outputs = model(noisy_images, t, labels)
                loss = criterion(outputs, noise)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            loader.set_description(f"Epoch [{epoch+1}/{args.num_epochs}] Loss: {avg_loss:.4f}")
        scheduler.step()

        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
                
        # Save the model checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'))
        
        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'model_best.pth'))

        # Save the latest model
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_latest.pth'))
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)    
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--n_channel', type=int, default=64)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--noise_schedule', type=str, default='linear', choices=['linear', 'scaled_linear','squaredcos_cap_v2', 'sigmoid', 'cosine'])

    parser.add_argument('--dataset', type=str, default='iclevr')
    parser.add_argument('--save_dir', type=str, default='cddpm_64')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to the checkpoint to load the model from')
    parser.add_argument('--start_epoch', type=int, default=0)

    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--save_interval', type=int, default=10)

    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision', default=False)
    args = parser.parse_args()

    train(args)