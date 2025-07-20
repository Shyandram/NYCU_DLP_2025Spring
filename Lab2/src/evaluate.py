import torch
from utils import dice_score

def evaluate(net, data, device, loss_fn = None, writer = None, epoch = None):
    # implement the evaluation function here
    net.eval()
    with torch.no_grad():
        val_loss = 0
        dice_score_val = 0
        for sample in data:
            images, masks = sample['image'], sample['mask']
            images, masks = images.to(device), masks.to(device)
            outputs = net(images)
            dice_score_val += dice_score(outputs, masks)
            val_loss += loss_fn(outputs, masks).item() if loss_fn else 0
        val_loss /= len(data)
        dice_score_val /= len(data)
        print(f"Validation Dice Score: {dice_score_val}")
        print(f"Validation Loss: {val_loss}") if loss_fn else None

        if writer:
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Dice/val', dice_score_val, epoch)
        
    return dice_score_val, val_loss
    
if __name__ == "__main__":
    pass