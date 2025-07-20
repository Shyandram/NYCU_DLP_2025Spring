import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid


from diffusers import DDPMScheduler

from ddpm import cDDPM
from dataset import iclevr

from evaluator import evaluation_model

def denormalize(tensor):
    """Denormalize a tensor image."""
    tensor = tensor.clone()
    tensor = tensor * 0.5 + 0.5  # Unnormalize to [0, 1]
    return tensor.clamp(0, 1)  # Clamp to [0, 1]

@torch.no_grad()
def test(args, mode='test'):
    # Set up the save directory
    save_dir = os.path.join(args.save_dir, mode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
            
    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up the dataset and dataloader
    test_dataset = iclevr(
        root=args.dataset, 
        image_size=args.image_size,
        mode=mode
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Set up the model
    model = cDDPM(
        sample_size=args.image_size,
        n_channel=args.n_channel,
        depth=args.depth,
    ).to(device)
    assert args.ckpt is not None, "Checkpoint path is required for testing."    
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    print(f"Loaded model from {args.ckpt}")

    eval_model = evaluation_model()

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.timesteps,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule=args.noise_schedule,
    )
    
    # testing loop
    loader = tqdm(test_loader)
    total_acc = 0.
    results = []
    for i, labels in enumerate(loader):
        images = torch.randn((labels.size(0), 3, args.image_size, args.image_size)).to(device)
        labels = labels.float().to(device)
        denosing_process = []
        for t in tqdm(noise_scheduler.timesteps):
            n_t = model(images, t, labels)            
            images = noise_scheduler.step(n_t, t, images).prev_sample

            if (t + 1) % args.save_interval == 0 or t == 0:
                denosing_process.append(denormalize(images).cpu())

        results.append(denormalize(images).squeeze(0).cpu())
        denosing_process = torch.stack(denosing_process, dim=0)
        denosing_process = denosing_process.reshape(-1, 3, args.image_size, args.image_size)
        save_image(denosing_process, os.path.join(save_dir, f"denosing_process_{mode}_{i}.png"), nrow=denosing_process.size(0))
        save_image(denormalize(images).cpu(), os.path.join(save_dir, f"final_{mode}_{i}.png"), nrow=denosing_process.size(0))

        acc = eval_model.eval(images, labels)
        total_acc += acc

        loader.set_postfix({"acc": total_acc/(i+1)})
    print(f"Mode: {mode} total accuracy: {total_acc / len(test_loader)}")
    
    make_grid_img = make_grid(torch.stack(results, dim=0), nrow=8)
    save_image(make_grid_img, os.path.join(save_dir, f"final_{mode}.png"))



        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--n_channel', type=int, default=64)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--noise_schedule', type=str, default='linear', choices=['linear', 'scaled_linear','squaredcos_cap_v2', 'sigmoid', 'cosine'])


    parser.add_argument('--dataset', type=str, default='iclevr')
    parser.add_argument('--save_dir', type=str, default='cddpm_64/samples')
    parser.add_argument('--ckpt', type=str, default='cddpm_64\model_best.pth')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--save_interval', type=int, default=100)
    args = parser.parse_args()

    test(args, mode='test')
    test(args, mode='new_test')