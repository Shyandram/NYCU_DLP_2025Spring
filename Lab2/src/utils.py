import torch
import torchvision as tv
import os

def dice_score(pred_mask, gt_mask, smooth = 1e-8):
    # implement the Dice score here
    assert pred_mask.shape == gt_mask.shape

    pred_mask_flat = pred_mask.view(-1)
    gt_mask_flat = gt_mask.view(-1)

    intersection = torch.sum(pred_mask_flat * gt_mask_flat)
    union = torch.sum(pred_mask_flat + gt_mask_flat)
    dice = (2 * intersection + smooth) / (union + smooth)
        
    return dice

def save_images(args, i, image, output, mask):
    # implement the save images function here
    save_path = os.path.join(args.save_path, args.model)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_img = torch.cat([denormalize(image), output.repeat(1, 3, 1, 1), mask.repeat(1, 3, 1, 1)], 0)
    tv.utils.save_image(save_img, os.path.join(save_path, f"output_{i}.png"), nrow=3)

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
    return tensor * std + mean