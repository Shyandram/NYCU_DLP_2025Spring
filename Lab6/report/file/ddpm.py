import torch
import torch.nn as nn
from diffusers import UNet2DModel

class cDDPM(nn.Module):
    def __init__(self, num_classes=24, n_channel=32, depth=4, sample_size=64):
        super().__init__()
        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=[n_channel * (i+1) for i in range(depth)],
            down_block_types=["DownBlock2D"] + ["AttnDownBlock2D"] * (depth - 1),
            up_block_types=["AttnUpBlock2D"] * (depth - 1) + ["UpBlock2D"],
            mid_block_type="UNetMidBlock2D",
            class_embed_type="identity",
            num_class_embeds=num_classes,
        )
        self.cls_emb = nn.Linear(num_classes, n_channel * depth)
    
    def forward(self, x, t, label):
        cls_emb = self.cls_emb(label)
        return self.unet(x, t, cls_emb).sample


if __name__ == "__main__":
    model = cDDPM()
    x = torch.randn(1, 3, 64, 64)
    t = torch.randint(0, 1000, (1,), dtype=torch.long)
    label = torch.randint(0, 1, (1, 24), dtype=torch.float)

    print(model)
    print(model(x, t, label).shape)