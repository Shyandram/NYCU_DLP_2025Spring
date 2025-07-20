import torch
import argparse

from models import UNet, ResNet34_UNet
from oxford_pet import load_dataset
from utils import save_images

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')

    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'resnet34_unet'], help='model to use')
    parser.add_argument('--ckpt', type=str, help='load model checkpoint', default=r'saved_models\train\deconv\unet\checkpoints\model_194_0.93.pth')
    # parser.add_argument('--model', type=str, default='resnet34_unet', choices=['unet', 'resnet34_unet'], help='model to use')
    # parser.add_argument('--ckpt', type=str, help='load model checkpoint', default=r'saved_models\train\deconv\resnet34_unet\checkpoints\model_151_0.91.pth')

    parser.add_argument('--data_path', type=str, default = './dataset', help='path to the input data')
    parser.add_argument('--save_path', type=str, default = './results/inference', help='path to save the output masks')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--upsample', type=str, default='Deconv', choices=['bilinear', 'Deconv'], help='upsample method')
    parser.add_argument('--use_gpu', action='store_true', help='use GPU', default=False)
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    
    return parser.parse_args()
            
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
    else:
        assert False, "Model path not provided!"
    return model

def inference(args, net, data, device):
    # implement the evaluation function here
    net.eval()
    with torch.no_grad():
        for i, sample in enumerate(data):
            image, mask = sample['image'], sample['mask']
            image, mask = image.to(device), mask.to(device)
            outputs = net(image)
            # Save the output masks
            save_images(args, i, image, outputs, mask)

if __name__ == '__main__':
    args = get_args()
    
    # Set the device
    device = torch.device(f"cuda:{args.gpu}" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model
    model = initialize_model(args, device)
    
    # Load the data
    data = load_dataset(args, mode='test')
    # Run the inference
    inference(args, model, data, device)
