import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
try:
    from .VQGAN import VQGAN
    from .Transformer import BidirectionalTransformer
except:
    from VQGAN import VQGAN
    from Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        codebook_mapping, codebook_indices, q_loss = self.vqgan.encode(x)
        return codebook_mapping, codebook_indices
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            def masking_ratio(ratio):
                return 1 - ratio
            return masking_ratio
        elif mode == "cosine":
            def masking_ratio(ratio):                
                return math.cos(math.pi * ratio * 0.5) 
            return masking_ratio
        elif mode == "square":
            def masking_ratio(ratio):
                return 1-ratio ** 2
            return masking_ratio
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x, ratio):

        z, z_indices = self.encode_to_z(x)
        z_indices = z_indices.view(-1, self.num_image_tokens)
        
        # Mask the tokens
        z_indices_input = self.apply_masking(ratio, z_indices)        

        logits = self.transformer(z_indices_input)
                        
        return logits, z_indices

    def apply_masking(self, ratio, z_indices):
        mask_token = torch.bernoulli(torch.ones_like(z_indices, ) * ratio)
        mask_token_id = torch.tensor(self.mask_token_id).to(z_indices.device)
        z_indices = torch.where(mask_token == 1, mask_token_id, z_indices)
        return z_indices
    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, z_indices, mask_b, ratio, mask_num, mask_func="cosine"):
        # raise Exception('TODO3 step1-1!')
        z_indices = torch.where(mask_b == 1, torch.tensor(self.mask_token_id).to(z_indices.device), z_indices)
        z_indices = z_indices.view(-1, self.num_image_tokens)
        logits = self.transformer(z_indices)
        #Apply softmax to convert logits into a probability distribution across the last dimension.
        logits = torch.nn.functional.softmax(logits, dim=-1)

        #FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = torch.max(logits, dim=-1)

        ratio = self.gamma_func(mask_func)(ratio)
        #predicted probabilities add temperature annealing gumbel noise as confidence
        
        # gumbel noise
        g = -torch.log(-torch.log(torch.rand_like(z_indices_predict_prob) + 1e-9) + 1e-9)
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g        
        
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        #sort the confidence for the rank 
        #define how much the iteration remain predicted tokens by mask scheduling
        ##At the end of the decoding process, add back the original(non-masked) token values
        mask_num = (mask_num * ratio).long()
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        cut_off = sorted_confidence[:,mask_num].unsqueeze(-1)
        mask_bc= (confidence < cut_off)
        return z_indices_predict, mask_bc
    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    

if __name__ == '__main__':
    model = MaskGit(yaml.safe_load(open('config/MaskGit.yml', 'r'))["model_param"])
    x = torch.randn(3, 3, 64, 64)
    mask_b = torch.randn(3, 768).bool()
    ratio = 0.5
    logits, z_indices = model(x, ratio)
    z_indices_predict, mask_bc = model.inpainting(x, mask_b, ratio)
    pass