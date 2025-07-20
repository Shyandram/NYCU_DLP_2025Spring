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

def test(args):

    # ---------------------------- Data and Model Preparation ----------------------------
    # Set the device
    device = torch.device(f"cuda:{args.gpu}" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the data
    val_loader = load_dataset(args, mode = 'test')
    print(f"Data mode: Test, Number of samples: {len(val_loader.dataset)}")

    # Initialize the model
    model = initialize_model(args, device)
    print(f"Model: {args.model}, Upsample: {args.upsample}")
    print(f'Checkpoint: {args.ckpt}')

    # ---------------------------- Testing Loop ----------------------------
    # test the model
    
    dice_score_val, val_loss = evaluate(model, val_loader, device)
    print(f"Dice Score: {dice_score_val:.4f}")

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
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        model.eval()
        print(f"Model loaded from {args.ckpt}")
    return model

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default = './dataset', help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3, help='learning rate')

    parser.add_argument('--project_name', '-n', type=str, default='', help='project name')

    # parser.add_argument('--model', type=str, default='unet', choices=['unet', 'resnet34_unet'], help='model to use')
    # parser.add_argument('--ckpt', type=str, help='load model checkpoint', default=r'saved_models\train\deconv\unet\checkpoints\model_194_0.93.pth')
    parser.add_argument('--model', type=str, default='resnet34_unet', choices=['unet', 'resnet34_unet'], help='model to use')
    parser.add_argument('--ckpt', type=str, help='load model checkpoint', default=r'saved_models\train\deconv\resnet34_unet\checkpoints\model_151_0.91.pth')

    parser.add_argument('--upsample', type=str, default='Deconv', choices=['bilinear', 'Deconv'], help='upsample method')

    parser.add_argument('--use_gpu', action='store_true', help='use GPU', default=True)
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--amp', action='store_true', help='use automatic mixed precision', default=False)

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    test(args)