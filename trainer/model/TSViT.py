import torch
from torch import nn, einsum
import torch.nn.functional as F

import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .module import *
#/workspace/tailing/trainer/model/module.py
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        """
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        """
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        

    def forward(self, x):
        return self.net(x)
        

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., return_last = False):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        #self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        #self.to_out = nn.Sequential(
        #    nn.Linear(inner_dim, dim),
        #    nn.Dropout(dropout)
        #) if project_out else nn.Identity()

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        
        qkv = self.to_qkv(x).chunk(3, dim = -1) #to_qkv
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)


        #qkv_attn
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out), attn

class ConvAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.in_depthwiseconv = SepConv2d(dim, dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attn_depthwiseconv = SepConv2d(dim, dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, d, h = *x.shape, self.heads

        cls = x[:, :1]
        image_token = x[:, 1:]
        H = W = int(math.sqrt(n - 1))

        image_token = rearrange(image_token, 'b (l w) d -> b d l w', l=H, w=W)
        image_token = self.in_depthwiseconv(image_token)
        image_token = rearrange(image_token, ' b d h w -> b (h w) d')
        x = x + torch.cat((cls, image_token), dim=1)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        k = k.transpose(2, 3)
        k = k.softmax(dim=-1)
        context = einsum('b h i j, b h j a -> b h i a', k, v)
        attn = einsum('b h i j, b h j j -> b h i j', q, context)

        cls = v[:, :, :1]
        value_token = v[:, :, 1:]
        value_token = rearrange(value_token, 'b h (l w) d -> b (h d) l w', l=H, w=W)
        value_token = self.attn_depthwiseconv(value_token)
        value_token = rearrange(value_token, ' b (h d) l w -> b h (l w) d', h=h)
        v = torch.cat((cls, value_token), dim=2)

        out = einsum('b h i j, b h i j -> b h i j', q, v)
        out = out + attn

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out, attn
    

class TrjAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.head_dim = dim // heads

        self.attend = nn.Softmax(dim = -1)
        #self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        self.proj_q = nn.Linear(dim, dim, bias=False)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.attn_drop = nn.Dropout(0.,)
        #self.to_out = nn.Sequential(
        #    nn.Linear(inner_dim, dim),
        #    nn.Dropout(dropout)
        #) if project_out else nn.Identity()

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        F = 16
        p = 196
        
        #qkv = self.to_qkv(x).chunk(3, dim = -1) #to_qkv
        #q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        q,k,v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q,k,v))
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
            lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

        #qkv_attn
        dots = einsum('b i d, b j d -> b i j', cls_q * self.scale, k) 
        attn = self.attend(dots)
        cls_out = einsum('b i j, b j d -> b i d', attn, v)
        cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h)
        
        x = orthoformer(
                q_, k_, v_,
                num_landmarks=128,
                num_frames=F,
            )
        
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=b)
        
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F)
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2)
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F)
        
        #temporal attention
        q2 = self.proj_q(x_diag)
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1)
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)
        attn = attn.softmax(dim=-1)
        
        x = rearrange(x, f'b s f (h d) -> b h s f d', f=F,  h=h)
        x = torch.einsum('b h s f, b h s f d -> b h s d', attn, x)
        x = rearrange(x, f'b h s d -> b s (h d)')
        x = torch.cat((cls_out, x), dim=1)
        
        return self.to_out(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                #PreNorm(dim, LeFF(dim = 192, scale = 4, depth_kernel = 3)),
                #PreNorm(dim, LCAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                #reNorm(dim, ReAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                #PreNorm(dim, ConvAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            attn_x, attn = attn(x)
            x = attn_x + x
            x = ff(x) + x
        return self.norm(x),  attn



# ViViT 모델 (모델 명만 안 바꿈)
class TSViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, num_frames, dim, depth, heads, mlp_dim, pool = 'cls', in_channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0.,):
        super().__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        
        #tube_dim = in_channels * tube_h * tube_w * tube_t
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )
        self.pretrain_pth = '/workspace/tailing/weights/pretrained/vit_small_patch16_224.pth'
        self.weights_from = "imagenet"
        self.depth = depth
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        #self.space_transformer = TransformerLeFF(dim, depth, heads, dim_head, scale = 4, depth_kernel = 3, dropout = 0.)
                                    
        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        #self.temporal_transformer = TransformerLeFF(dim, depth, heads, dim_head, scale = 4, depth_kernel = 3, dropout = 0.)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.init_weights()
        self.dim = dim
        self.inner_dim = dim * dim_head

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x) #torch.Size([1, 16, 17, 768])
        
        x = rearrange(x, 'b t n d -> (b t) n d')
        x , _ = self.space_transformer(x)

        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x, _ = self.temporal_transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)
    
    # vit pretrained weight 불러오는 것
    def init_weights(self):
		
        #if self.use_learnable_pos_emb:
			#trunc_normal_(self.pos_embed, std=.02)
			#trunc_normal_(self.time_embed, std=.02)
		#	nn.init.trunc_normal_(self.pos_embed, std=.02)
	    #	nn.init.trunc_normal_(self.time_embed, std=.02)
		#trunc_normal_(self.cls_token, std=.02)
		
        if self.pretrain_pth is not None:
            if self.weights_from == 'imagenet':
                init_from_vit_pretrain_(self,
                                        pretrained = self.pretrain_pth,
                                        conv_type=None,
                                        attention_type = 'fact_encoder',
                                        copy_strategy='repeat',
                                        extend_strategy='temporal_avg', 
                                        tube_size = 2, 
                                        num_time_transformer_layers = self.depth)
                
    def get_last_selfattention(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b = b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x) #torch.Size([1, 16, 17, 768])

        x = rearrange(x, 'b t n d -> (b t) n d')
        x,attn= self.space_transformer(x)
        """
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        
        x , attn = self.temporal_transformer(x)
        """
        return attn


    


    