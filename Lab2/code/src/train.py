import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter

from tqdm import tqdm
import argparse
import os

from models import UNet, ResNet34_UNet
from oxford_pet import load_dataset
from evaluate import evaluate

def train(args):

    # ---------------------------- Data and Model Preparation ----------------------------
    # Set the device
    device = torch.device(f"cuda:{args.gpu}" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    enable_amp = args.amp and device.type == 'cuda'
    scaler = GradScaler(enabled=enable_amp)

    # Set the save folder
    model_dir = os.path.join('./saved_models/train')
    if args.project_name:
        model_dir = os.path.join(model_dir, args.project_name, args.model)
    else:
        model_dir = os.path.join(model_dir, args.model)
    ckpt_dir = os.path.join(model_dir, 'checkpoints')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)


    # Load the data
    train_loader = load_dataset(args, mode = 'train')
    val_loader = load_dataset(args, mode = 'valid')

    # Initialize the model
    model = initialize_model(args, device)
    
    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    # Initialize the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min = 1e-6)
    # Initialize the loss function
    loss_fn = nn.BCEWithLogitsLoss(size_average=True)

    # Tensorboard
    writer = SummaryWriter(os.path.join(model_dir, 'logs'))

    # ---------------------------- Training Loop ----------------------------
    best_dice_score = 0
    # Train the model
    print("Start training...")
    for epoch in range(args.epochs):
        train_loss = 0
        model.train()
        train_loop = tqdm(train_loader)
        for batch, sample in enumerate(train_loop):
            optimizer.zero_grad()

            inputs, masks = sample['image'], sample['mask']
            inputs, masks = inputs.to(device), masks.to(device)

            with autocast(enabled=enable_amp):
                outputs = model(inputs)
                loss = loss_fn(outputs, masks)

            # loss.backward()
            # optimizer.step()
            # scheduler.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            train_loop.set_description(f"Epoch [{epoch}/{args.epochs}]")
            train_loop.set_postfix(loss = train_loss / (batch + 1))

        writer.add_scalar('Loss/train', loss.item(), epoch)

        # Validation
        dice_score_val, val_loss = evaluate(model, val_loader, device, loss_fn, writer, epoch)

        # Save the model
        if dice_score_val > best_dice_score:
            best_dice_score = dice_score_val
            ckpt_path = os.path.join(ckpt_dir, f"model_best.pth")
            # ckpt_path = os.path.join(ckpt_dir, f"model_{epoch}_{best_dice_score:.2f}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Model saved to {ckpt_path}")

def initialize_model(args, device):
    class_num = 1
    if args.model == 'unet':
        model = UNet(in_channels=3, out_channels=class_num, upsample=args.upsample)
        print("Using UNet model")
    elif args.model == 'resnet34_unet':
        model = ResNet34_UNet(in_channels=3, out_channels=class_num, upsample=args.upsample)
        print("Using resnet34_unet model")
    else:
        assert False, "Unknown model!"
    
    model.to(device)

    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt))
        print(f"Model loaded from {args.ckpt}")
    return model

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default = './dataset', help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3, help='learning rate')

    parser.add_argument('--project_name', '-n', type=str, default='', help='project name')

    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'resnet34_unet'], help='model to use')
    parser.add_argument('--ckpt', type=str, help='load model checkpoint')
    parser.add_argument('--upsample', type=str, default='Deconv', choices=['bilinear', 'Deconv'], help='upsample method')

    parser.add_argument('--use_gpu', action='store_true', help='use GPU', default=False)
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--amp', action='store_true', help='use automatic mixed precision', default=False)

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)