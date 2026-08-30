"""Faithful, dependency-light port of the published RDAH-Net inference graph.

The structure and layer names deliberately follow the authors' public
``test.py`` so the Figshare Track1 state dictionary can be checked with strict
loading.  This file has no dataset or plotting code from the research repo.
"""

from __future__ import annotations

import math


def build_rdah_net():
    """Construct the official graph lazily, keeping geospatial tooling torch-free."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ConvLayer(nn.Module):
        def __init__(self, i, o, kernel_size=3, stride=1, padding=1, groups=1, act=True):
            super().__init__(); self.conv = nn.Conv2d(i, o, kernel_size, stride, padding, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(o); self.act = nn.GELU() if act else nn.Identity()
        def forward(self, x): return self.act(self.bn(self.conv(x)))

    class BlockAttention(nn.Module):
        def __init__(self, dim, num_heads=4, block_size=8, mlp_dim=None, dropout=0.):
            super().__init__(); assert dim % num_heads == 0
            self.num_heads, self.head_dim, self.block_size = num_heads, dim // num_heads, block_size
            self.scale = self.head_dim ** -.5; self.local_proj = ConvLayer(dim, dim, groups=dim)
            self.qkv = nn.Conv2d(dim, dim * 3, 1); self.attn_drop = nn.Dropout(dropout)
            self.proj = nn.Conv2d(dim, dim, 1); self.proj_drop = nn.Dropout(dropout)
            self.mlp = nn.Sequential(nn.Conv2d(dim, mlp_dim or dim * 2, 1), nn.GELU(), nn.Dropout(dropout),
                                     nn.Conv2d(mlp_dim or dim * 2, dim, 1), nn.Dropout(dropout))
        def forward(self, x):
            b, c, h, w = x.shape
            if h % self.block_size or w % self.block_size: raise ValueError("RDAH input tile must remain divisible by 128")
            bh, bw = h // self.block_size, w // self.block_size
            local = self.local_proj(x)
            blocked = x.reshape(b, c, bh, self.block_size, bw, self.block_size).permute(0,2,4,1,3,5).reshape(b*bh*bw,c,self.block_size,self.block_size)
            bb, _, hh, ww = blocked.shape; n = hh * ww
            qkv = self.qkv(blocked).reshape(bb,3,c,n).permute(1,0,2,3)
            q, k, v = (z.reshape(bb,self.num_heads,self.head_dim,n) for z in qkv)
            attention = self.attn_drop((torch.einsum("bhdn,bhdm->bhnm",q,k)*self.scale).softmax(-1))
            out = torch.einsum("bhnm,bhdm->bhdn",attention,v).reshape(bb,c,hh,ww)
            out = self.proj_drop(self.proj(out)).reshape(b,bh,bw,c,hh,ww).permute(0,3,1,4,2,5).reshape(b,c,h,w)
            x = x + local + out
            return x + self.mlp(x)

    class MobileViTBlock(nn.Module):
        def __init__(self, i, o, stride=1, num_heads=4):
            super().__init__(); self.conv1 = ConvLayer(i,o,stride=stride); self.attention=BlockAttention(o,num_heads); self.conv2=ConvLayer(o,o,groups=o)
        def forward(self,x): return self.conv2(self.attention(self.conv1(x)))

    class MobileViT_S_Light(nn.Module):
        def __init__(self, channels=3):
            super().__init__(); self.stem=ConvLayer(channels,32,kernel_size=4,stride=2,padding=1)
            self.stage1=nn.Sequential(MobileViTBlock(32,64,2,4),MobileViTBlock(64,64,1,4))
            self.stage2=nn.Sequential(MobileViTBlock(64,128,2,8),MobileViTBlock(128,128,1,8))
            self.stage3=nn.Sequential(MobileViTBlock(128,256,2,8),MobileViTBlock(256,256,1,8))
            self.proj1=ConvLayer(64,32,kernel_size=1,padding=0); self.proj2=ConvLayer(128,32,kernel_size=1,padding=0); self.proj3=ConvLayer(256,32,kernel_size=1,padding=0)
        def forward(self,x):
            x=self.stem(x); f1=self.stage1(x); f2=self.stage2(f1); f3=self.stage3(f2)
            return [self.proj1(f1),self.proj2(f2),self.proj3(f3)]

    class CBAM(nn.Module):
        def __init__(self, channels, reduction=16):
            super().__init__(); self.avg_pool=nn.AdaptiveAvgPool2d(1); self.max_pool=nn.AdaptiveMaxPool2d(1)
            self.fc=nn.Sequential(nn.Conv2d(channels,channels//reduction,1,bias=False),nn.ReLU(),nn.Conv2d(channels//reduction,channels,1,bias=False))
            self.spatial=nn.Conv2d(2,1,7,padding=3,bias=False); self.sigmoid=nn.Sigmoid()
        def forward(self,x):
            x=x*self.sigmoid(self.fc(self.avg_pool(x))+self.fc(self.max_pool(x)))
            return x*self.sigmoid(self.spatial(torch.cat((x.mean(1,keepdim=True),x.max(1,keepdim=True)[0]),1)))

    class PositionalEncoding(nn.Module):
        def __init__(self,d_model=32,h=64,w=64):
            super().__init__(); px=torch.arange(w,dtype=torch.float32).repeat(h,1); py=torch.arange(h,dtype=torch.float32).repeat(w,1).t()
            pe=torch.zeros(1,d_model,h,w); term=torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.)/d_model))
            pe[0,::2]=torch.sin(px[None]*term[:,None,None]); pe[0,1::2]=torch.cos(py[None]*term[:,None,None]); self.register_buffer("pe",pe)
        def forward(self,x): return x+self.pe[:,:x.size(1),:x.size(2),:x.size(3)]

    class LightCrossAttention(nn.Module):
        def __init__(self,d_model=32,num_heads=4,block_size=8,dropout=.1):
            super().__init__(); self.num_heads=num_heads; self.head_dim=d_model//num_heads; self.block_size=block_size; self.scale=self.head_dim**-.5
            self.dropout=nn.Dropout(dropout); self.proj_q=nn.Conv2d(d_model,d_model,1); self.proj_k=nn.Conv2d(d_model,d_model,1); self.proj_v=nn.Conv2d(d_model,d_model,1); self.proj_out=nn.Conv2d(d_model,d_model,1); self.norm=nn.BatchNorm2d(d_model)
        def forward(self,q,k,v):
            b,c,h,w=q.shape; bh,bw=h//self.block_size,w//self.block_size
            def blocks(x): return x.reshape(b,c,bh,self.block_size,bw,self.block_size).permute(0,2,4,1,3,5).reshape(b*bh*bw,c,self.block_size,self.block_size)
            original=q; q,k,v=blocks(q),blocks(k),blocks(v); bb,_,hh,ww=q.shape; n=hh*ww
            q,k,v=(layer(x).reshape(bb,self.num_heads,self.head_dim,n) for layer,x in ((self.proj_q,q),(self.proj_k,k),(self.proj_v,v)))
            attn=self.dropout(F.softmax(torch.einsum("bhdn,bhdm->bhnm",q,k)*self.scale,-1))
            out=torch.einsum("bhnm,bhdm->bhdn",attn,v).reshape(bb,c,hh,ww); out=self.proj_out(out)
            out=out.reshape(b,bh,bw,c,hh,ww).permute(0,3,1,4,2,5).reshape(b,c,h,w)
            return self.norm(out+original)

    class LightTransformerBlock(nn.Module):
        def __init__(self,d_model=32,num_heads=4,hidden_dim=64,block_size=8,dropout=.1):
            super().__init__(); self.self_attn=LightCrossAttention(d_model,num_heads,block_size,dropout)
            self.ffn=nn.Sequential(nn.Conv2d(d_model,hidden_dim,1),nn.ReLU(),nn.Dropout(dropout),nn.Conv2d(hidden_dim,d_model,1)); self.norm=nn.BatchNorm2d(d_model); self.dropout=nn.Dropout(dropout)
        def forward(self,x): x=self.self_attn(x,x,x); return x+self.dropout(self.ffn(self.norm(x)))

    class HeightPredTransformer(nn.Module):
        def __init__(self,d_model=32,num_heads=4):
            super().__init__(); self.depth_encoder=MobileViT_S_Light(1); self.img_encoder=MobileViT_S_Light(3)
            self.cbam_blocks=nn.ModuleList([CBAM(d_model) for _ in range(3)])
            self.cross_attn_blocks=nn.ModuleList([LightCrossAttention(d_model,num_heads) for _ in range(3)])
            self.rev_cross_attn_blocks=nn.ModuleList([LightCrossAttention(d_model,num_heads) for _ in range(3)])
            self.pos_encoding=PositionalEncoding(d_model,64,64); self.global_transformer=LightTransformerBlock(d_model,num_heads,64)
            self.skip_projs=nn.ModuleList([nn.Conv2d(d_model,16,1),nn.Conv2d(d_model,32,1),nn.Conv2d(d_model,64,1)])
            self.decoder=nn.Sequential(nn.Conv2d(d_model,256,3,padding=1),nn.PixelShuffle(2),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,128,3,padding=1),nn.PixelShuffle(2),nn.BatchNorm2d(32),nn.ReLU(),nn.Conv2d(32,64,3,padding=1),nn.PixelShuffle(2),nn.BatchNorm2d(16),nn.ReLU(),nn.Conv2d(16,32,3,padding=1),nn.PixelShuffle(2),nn.BatchNorm2d(8),nn.ReLU(),nn.Conv2d(8,1,3,padding=1))
        def forward(self,depth,img):
            df=[self.cbam_blocks[i](f) for i,f in enumerate(self.depth_encoder(depth))]; im=[self.cbam_blocks[i](f) for i,f in enumerate(self.img_encoder(img))]
            fused=[(self.cross_attn_blocks[i](df[i],im[i],im[i])+self.rev_cross_attn_blocks[i](im[i],df[i],df[i]))/2 for i in range(3)]
            x=self.global_transformer(self.pos_encoding(fused[2])); x=self.decoder[:4](x); x=x+F.interpolate(self.skip_projs[2](fused[2]),x.shape[2:],mode="bilinear",align_corners=False)
            x=self.decoder[4:8](x); x=x+F.interpolate(self.skip_projs[1](fused[1]),x.shape[2:],mode="bilinear",align_corners=False)
            x=self.decoder[8:12](x); x=x+F.interpolate(self.skip_projs[0](fused[0]),x.shape[2:],mode="bilinear",align_corners=False)
            return self.decoder[12:](x)

    return HeightPredTransformer()
