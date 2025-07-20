import torch.nn as nn
import torch
import math

#TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3)
        self.linear = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        '''
        qkv = self.qkv(x) # shape = [batch_size, num_image_tokens, 3*dim]
        qkv = qkv.reshape(x.shape[0], x.shape[1], 3, self.num_heads, self.head_dim) # shape = [batch_size, num_image_tokens, 3, num_heads, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # shape = [batch_size, num_heads, num_image_tokens, head_dim]
        
        attn = self.scale * (q @ k.transpose(-2, -1)) # shape = [batch_size, num_heads, num_image_tokens, num_image_tokens]
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        attn = attn @ v # shape = [batch_size, num_heads, num_image_tokens, head_dim]
        attn = attn.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], self.dim) # shape = [batch_size, num_image_tokens, dim]
        return self.linear(attn)


class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    

if __name__ == '__main__':
    # Test MultiHeadAttention
    x = torch.randn(2, 10, 768)
    mha = MultiHeadAttention()
    print(mha(x).shape)
    
    # Test MLP
    mlp = MLP()
    print(mlp(x).shape)
    
    # Test TokenPredictor
    tp = TokenPredictor()
    print(tp(x).shape)
    
    # Test Encoder
    encoder = Encoder()
    print(encoder(x).shape)