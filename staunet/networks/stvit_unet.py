
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops.layers.torch import Rearrange
import torch.utils.checkpoint as checkpoint
import numpy as np

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., conv_pos=True,
                 downsample=False, kernel_size=5):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features = out_features or in_features
        self.hidden_features = hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act1 = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

        self.conv = ResDWC(hidden_features, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.conv(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, window_size=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.window_size = window_size

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        q, k, v = self.qkv(x).reshape(B, self.num_heads, C // self.num_heads * 3, N).chunk(3,
                                                                                           dim=2)  # (B, num_heads, head_dim, N)

        attn = (k.transpose(-1, -2) @ q) * self.scale

        attn = attn.softmax(dim=-2)  # (B, h, N, N)
        attn = self.attn_drop(attn)

        x = (v @ attn).reshape(B, C, H, W)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Unfold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        weights = torch.eye(kernel_size ** 2)
        weights = weights.reshape(kernel_size ** 2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.shape
        x = F.conv2d(x.reshape(b * c, 1, h, w), self.weights, stride=1, padding=self.kernel_size // 2)
        return x.reshape(b, c * 9, h * w)


class Fold(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size

        weights = torch.eye(kernel_size ** 2)
        weights = weights.reshape(kernel_size ** 2, 1, kernel_size, kernel_size)
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        b, _, h, w = x.shape
        x = F.conv_transpose2d(x, self.weights, stride=1, padding=self.kernel_size // 2)
        return x

class StokenAttention(nn.Module):
    def __init__(self, dim, stoken_size, seg_layers=1, hard_label=True, refine=True, refine_attention=True,
                 num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., rpe=False):
        super().__init__()

        self.n_iter = seg_layers
        self.stoken_size = stoken_size

        self.refine = refine
        self.refine_attention = refine_attention

        self.scale = dim ** - 0.5

        self.unfold = Unfold(3)
        self.fold = Fold(3)

        if refine:

            if refine_attention:
                self.stoken_refine = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                               attn_drop=attn_drop, proj_drop=proj_drop)
            else:
                self.stoken_refine = nn.Sequential(
                    nn.Conv2d(dim, dim, 1, 1, 0),
                    nn.Conv2d(dim, dim, 5, 1, 2, groups=dim),
                    nn.Conv2d(dim, dim, 1, 1, 0)
                )

    def stoken_forward(self, x):
        '''
           x: (B, C, H, W)
        '''
        B, C, H0, W0 = x.shape
        h, w = self.stoken_size

        pad_l = pad_t = 0
        pad_r = (w - W0 % w) % w
        pad_b = (h - H0 % h) % h
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))

        _, _, H, W = x.shape

        hh, ww = H // h, W // w


        stoken_features = F.adaptive_avg_pool2d(x, (hh, ww))  # (B, C, hh, ww)


        pixel_features = x.reshape(B, C, hh, h, ww, w).permute(0, 2, 4, 3, 5, 1).reshape(B, hh * ww, h * w, C)

        with torch.no_grad():
            for idx in range(self.n_iter):
                stoken_features = self.unfold(stoken_features)  # (B, C*9, hh*ww)
                stoken_features = stoken_features.transpose(1, 2).reshape(B, hh * ww, C, 9)
                affinity_matrix = pixel_features @ stoken_features * self.scale  # (B, hh*ww, h*w, 9)
  
                affinity_matrix = affinity_matrix.softmax(-1)  # (B, hh*ww, h*w, 9)
               
                affinity_matrix_sum = affinity_matrix.sum(2).transpose(1, 2).reshape(B, 9, hh, ww)
            
                affinity_matrix_sum = self.fold(affinity_matrix_sum)
                if idx < self.n_iter - 1:
                    stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix  # (B, hh*ww, C, 9)
                    
                    stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B * C, 9, hh, ww)).reshape(
                        B, C, hh, ww)
                 

                   
                    stoken_features = stoken_features / (affinity_matrix_sum + 1e-12)  # (B, C, hh, ww)
                

        stoken_features = pixel_features.transpose(-1, -2) @ affinity_matrix  # (B, hh*ww, C, 9)
        
        stoken_features = self.fold(stoken_features.permute(0, 2, 3, 1).reshape(B * C, 9, hh, ww)).reshape(B, C, hh, ww)

        stoken_features = stoken_features / (affinity_matrix_sum.detach() + 1e-12)  # (B, C, hh, ww)
       

        if self.refine:
            if self.refine_attention:
               
                stoken_features = self.stoken_refine(stoken_features)
                
            else:
                stoken_features = self.stoken_refine(stoken_features)



        stoken_features = self.unfold(stoken_features)  # (B, C*9, hh*ww)
        stoken_features = stoken_features.transpose(1, 2).reshape(B, hh * ww, C, 9)  # (B, hh*ww, C, 9)

        pixel_features = stoken_features @ affinity_matrix.transpose(-1, -2)  # (B, hh*ww, C, h*w)

        pixel_features = pixel_features.reshape(B, hh, ww, C, h, w).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)


        if pad_r > 0 or pad_b > 0:
            pixel_features = pixel_features[:, :, :H0, :W0]

        return pixel_features

    def direct_forward(self, x):
        B, C, H, W = x.shape
        stoken_features = x
        if self.refine:
            if self.refine_attention:
                # stoken_features = stoken_features.flatten(2).transpose(-1, -2)
                stoken_features = self.stoken_refine(stoken_features)
                # stoken_features = stoken_features.transpose(-1, -2).reshape(B, C, H, W)
            else:
                stoken_features = self.stoken_refine(stoken_features)
        return stoken_features

    def forward(self, x):
        if self.stoken_size[0] > 1 or self.stoken_size[1] > 1:
            return self.stoken_forward(x)
        else:
            return self.direct_forward(x)


class ResDWC(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super().__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim)


    def forward(self, x):

        return x + self.conv(x)



class StokenAttentionLayer(nn.Module):
    def __init__(self, dim, seg_layers, stoken_size, stoken_refine=True, stoken_refine_attention=True,
                 hard_label=False,
                 num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, layerscale=False, init_values=1.0e-5, rpe=False):
        super().__init__()

        self.layerscale = layerscale

        
        self.pos_embed = ResDWC(dim, 3)

        self.norm1 = LayerNorm2d(dim)
        
        self.attn = StokenAttention(dim,
                                    stoken_size=stoken_size, seg_layers=seg_layers,
                                    num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    attn_drop=attn_drop, proj_drop=drop, rpe=rpe)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, act_layer=act_layer,
                        drop=drop)

        if layerscale:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(1, dim, 1, 1), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(1, dim, 1, 1), requires_grad=True)

    def forward(self, x):
        x = self.pos_embed(x)

        if self.layerscale:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp2(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp2(self.norm2(x)))
        x_kd = x
        return x, x_kd



class PatchEmbed(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(out_channels // 2),

            nn.Conv2d(out_channels // 2, out_channels // 2, 3, 1, 1),
            nn.GELU(),
            nn.BatchNorm2d(out_channels // 2),

            nn.Conv2d(out_channels // 2, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(out_channels),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GELU(),
            nn.BatchNorm2d(out_channels),

        )

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x

class CARAFE(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=3, up_factor=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(dim, dim // 4, 1)
        self.encoder = nn.Conv2d(dim // 4, self.up_factor ** 2 * self.kernel_size ** 2,
                                 self.kernel_size, 1, self.kernel_size // 2)
        self.out = nn.Conv2d(dim, dim_out, 1)

    def forward(self, x):
        # B, new_HW, C = x.shape
        B, C, H, W = x.shape



            # N,C,H,W -> N,C,delta*H,delta*W
            # kernel prediction module
        kernel_tensor = self.down(x)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor,
                                        self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(B, self.kernel_size ** 2, H, W,
                                                  self.up_factor ** 2)  # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

            # content-aware reassembly module
            # tensor.unfold: dim, size, step
        w = F.pad(x, pad=(self.kernel_size // 2, self.kernel_size // 2,
                                              self.kernel_size // 2, self.kernel_size // 2),
                              mode='constant', value=0)  # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        w = w.unfold(2, self.kernel_size, step=1)  # (N, C, H, W+Kup//2+Kup//2, Kup)
        w = w.unfold(3, self.kernel_size, step=1)  # (N, C, H, W, Kup, Kup)
        w = w.reshape(B, C, H, W, -1)  # (N, C, H, W, Kup^2)
        w = w.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        x = torch.matmul(w, kernel_tensor)  # (N, H, W, C, S^2)
        x = x.reshape(B, H, W, -1)
        x = x.permute(0, 3, 1, 2)
        x = F.pixel_shuffle(x, self.up_factor)
        x = self.out(x)


        return x


class CARAFE4(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=3, up_factor=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(dim, dim // 4, 1)
        self.encoder = nn.Conv2d(dim // 4, self.up_factor ** 2 * self.kernel_size ** 2,
                                 self.kernel_size, 1, self.kernel_size // 2)
        self.out = nn.Conv2d(dim, dim_out, 1)

    def forward(self, x):
        # B, new_HW, C = x.shape
        B, C, H, W = x.shape

        kernel_tensor = self.down(x)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor,
                                        self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(B, self.kernel_size ** 2, H, W,
                                                  self.up_factor ** 2)  # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)


        w = F.pad(x, pad=(self.kernel_size // 2, self.kernel_size // 2,
                                              self.kernel_size // 2, self.kernel_size // 2),
                              mode='constant', value=0)  # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        w = w.unfold(2, self.kernel_size, step=1)  # (N, C, H, W+Kup//2+Kup//2, Kup)
        w = w.unfold(3, self.kernel_size, step=1)  # (N, C, H, W, Kup, Kup)
        w = w.reshape(B, C, H, W, -1)  # (N, C, H, W, Kup^2)
        w = w.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        x = torch.matmul(w, kernel_tensor)  # (N, H, W, C, S^2)
        x = x.reshape(B, H, W, -1)
        x = x.permute(0, 3, 1, 2)
        x = F.pixel_shuffle(x, self.up_factor)
        x = self.out(x)

        return x


class STViTUnetsys(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=[64, 128, 320, 512],
                 depth=[2, 2, 6, 1],
                 num_heads=[1, 2, 5, 8],
                 # seg_dim=[16, 16, 16, 16],
                 seg_layers=[1, 1, 1, 1],
                 stoken_size=[8, 4, 1, 1],               #？
                 mlp_ratio=4,
                 stoken_refine=True,
                 stoken_refine_attention=True,
                 hard_label=False,
                 rpe=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop=0.,
                 drop_path_rate=0,
                 projection=None,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 layerscale=[False, False, False, False],
                 init_values=1e-6,
                 freeze_bn=False,
                 use_chk=False):
        super().__init__()
        self.use_chk = use_chk
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_features = embed_dim[0]  # num_features for consistency with other models
        heads = num_heads
        # print("num_heads",num_heads)
        self.mlp_ratio = mlp_ratio
        self.freeze_bn = freeze_bn
        self.swish = MemoryEfficientSwish()
        #encoder

        self.patch_embed = PatchEmbed(in_chans, embed_dim[0])


        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule

        print("depth",depth)

        self.stage1 = nn.ModuleList(
            [StokenAttentionLayer(
                dim=embed_dim[0], num_heads=heads[0],seg_layers=seg_layers[0], stoken_size=to_2tuple(stoken_size[0]),
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop, act_layer=act_layer,
                layerscale=layerscale[0], init_values=init_values, rpe=rpe,
                drop_path=dpr[i])
            for i in range(depth[0])])

        self.merge1 = PatchMerging(embed_dim[0], embed_dim[1])


        self.stage2 = nn.ModuleList(
            [StokenAttentionLayer(
                dim=embed_dim[1], num_heads=heads[1], seg_layers=seg_layers[1], stoken_size=to_2tuple(stoken_size[1]),
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop, act_layer=act_layer,
                layerscale=layerscale[1], init_values=init_values, rpe=rpe,
                drop_path=dpr[np.sum(depth[:1]) + i])
                for i in range(depth[1])])
        self.merge2 = PatchMerging(embed_dim[1], embed_dim[2])



        self.stage3 = nn.ModuleList(
            [StokenAttentionLayer(
                dim=embed_dim[2], num_heads=heads[2], seg_layers=seg_layers[2], stoken_size=to_2tuple(stoken_size[2]),
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop, act_layer=act_layer,
                layerscale=layerscale[2], init_values=init_values, rpe=rpe,
                drop_path=dpr[np.sum(depth[:2]) + i])
                for i in range(depth[2])])
        self.merge3 = PatchMerging(embed_dim[2], embed_dim[3])


        self.stage4 = nn.ModuleList(
            [StokenAttentionLayer(
                dim=embed_dim[3], num_heads=heads[3], seg_layers=seg_layers[3], stoken_size=to_2tuple(stoken_size[3]),
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop, act_layer=act_layer,
                layerscale=layerscale[3], init_values=init_values, rpe=rpe,
                drop_path=dpr[np.sum(depth[:-1]) + i])
                for i in range(depth[-1])])



        self.stage_up4 = nn.ModuleList(
            [StokenAttentionLayer(
                dim=embed_dim[3], num_heads=heads[3], seg_layers=seg_layers[3], stoken_size=to_2tuple(stoken_size[3]),
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop, act_layer=act_layer,
                layerscale=layerscale[3], init_values=init_values, rpe=rpe,
                drop_path=dpr[np.sum(depth[:-1]) + i])
                for i in range(depth[-1])])

        self.upsample4 = CARAFE(embed_dim[3], embed_dim[2])
        

        self.concat_linear4 = nn.Linear(640, 320)

        self.stage_up3 = nn.ModuleList(
            [StokenAttentionLayer(
                dim=embed_dim[2], num_heads=heads[2], seg_layers=seg_layers[2], stoken_size=to_2tuple(stoken_size[2]),
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop, act_layer=act_layer,
                layerscale=layerscale[2], init_values=init_values, rpe=rpe,
                drop_path=dpr[np.sum(depth[:2]) + i])
                for i in range(depth[2])]
        )

        self.upsample3 = CARAFE(embed_dim[2], embed_dim[1])
        # self.upsample3 = PatchUpsampling(embed_dim[2], embed_dim[1])

        self.concat_linear3 = nn.Linear(256, 128)

        self.stage_up2 = nn.ModuleList(
            [StokenAttentionLayer(
                dim=embed_dim[1], num_heads=heads[1], seg_layers=seg_layers[1], stoken_size=to_2tuple(stoken_size[1]),
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop, act_layer=act_layer,
                layerscale=layerscale[1], init_values=init_values, rpe=rpe,
                drop_path=dpr[np.sum(depth[:1]) + i])
                for i in range(depth[1])])
        self.upsample2 = CARAFE(embed_dim[1], embed_dim[0])
        

        self.concat_linear2 = nn.Linear(128, 64)
        self.stage_up1 = nn.ModuleList([
            StokenAttentionLayer(
                dim=embed_dim[0], num_heads=heads[0], seg_layers=seg_layers[0], stoken_size=to_2tuple(stoken_size[0]),
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop, act_layer=act_layer,
                layerscale=layerscale[0], init_values=init_values, rpe=rpe,
                drop_path=dpr[i])
            for i in range(depth[0])])


        self.upsample1 = CARAFE4(embed_dim[0], 64)
        # self.upsample1 = PatchUpsampling4(embed_dim[0], 64)

        self.proj = nn.Conv2d(self.num_features, projection, 1) if projection else None


        self.norm1 = nn.BatchNorm2d(512)  #encoder
        self.norm = nn.BatchNorm2d(1024)  #decoder



        self.swish = MemoryEfficientSwish()

        self.concat_linear1 = nn.Linear(1024, 64)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.norm_up = norm_layer(embed_dim)
        self.output = nn.Conv2d(in_channels=embed_dim[0], out_channels=self.num_classes, kernel_size=1, bias=False)#分割头
        # Classifier head

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    #Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        kd_feature = []  

        # Stage 1
        for blk in self.stage1:
            if self.use_chk:
                x, fe = checkpoint.checkpoint(blk, x)
            else:
                x, fe = blk(x)
        self.x1 = x 
        kd_feature.append(fe)  
        x = self.merge1(x)

        # Stage 2
        for blk in self.stage2:
            if self.use_chk:
                x, fe = checkpoint.checkpoint(blk, x)
            else:
                x, fe = blk(x)
        self.x2 = x  
        kd_feature.append(fe) 
        x = self.merge2(x)

        # Stage 3
        for blk in self.stage3:
            if self.use_chk:
                x, fe = checkpoint.checkpoint(blk, x)
            else:
                x, fe = blk(x)
        self.x3 = x  
        kd_feature.append(fe)  
        x = self.merge3(x)

        # Stage 4
        for blk in self.stage4:
            if self.use_chk:
                x, fe = checkpoint.checkpoint(blk, x)
            else:
                x, fe = blk(x)
        kd_feature.append(fe) 
        
        x = self.norm1(x)

        return x, kd_feature

    #Dencoder and Skip connection
    def forward_up_features(self, x):

        
        kd_feature = []
        # Decoder Stage Up 4
        for blk in self.stage_up4:
            if self.use_chk:
                x, fe = checkpoint.checkpoint(blk, x)
            else:
                x, fe = blk(x)
        kd_feature.append(fe)  
        x = self.upsample4(x)

        x = torch.cat([self.x3, x], 1)

        # 处理 x 形状
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.view(B, H * W, C)
        x = self.concat_linear4(x)
        x = x.view(B, H, W, -1)
        x = x.permute(0, 3, 1, 2)

        # Decoder Stage Up 3
        for blk in self.stage_up3:
            if self.use_chk:
                x, fe = checkpoint.checkpoint(blk, x)
            else:
                x, fe = blk(x)
        kd_feature.append(fe)  
        x = self.upsample3(x)
        x = torch.cat([self.x2, x], 1)

        # 处理 x 形状
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.view(B, H * W, C)
        x = self.concat_linear3(x)
        x = x.view(B, H, W, -1)
        x = x.permute(0, 3, 1, 2)

        # Decoder Stage Up 2
        for blk in self.stage_up2:
            if self.use_chk:
                x, fe = checkpoint.checkpoint(blk, x)
            else:
                x, fe = blk(x)
        kd_feature.append(fe)  
        x = self.upsample2(x)
        x = torch.cat([self.x1, x], 1)

        
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.view(B, H * W, C)
        x = self.concat_linear2(x)
        x = x.view(B, H, W, -1)
        x = x.permute(0, 3, 1, 2)

        # Decoder Stage Up 1
        for blk in self.stage_up1:
            if self.use_chk:
                x, fe = checkpoint.checkpoint(blk, x)
            else:
                x, fe = blk(x)

        kd_feature.append(fe)

        x = self.proj(x)
        x = self.norm(x)
        x = self.swish(x)

        return x, kd_feature

    def up_x4(self, x):

        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1)
        x = x.view(B, H * W, C)
        x = self.concat_linear1(x)
        x = x.view(B, H, W, -1)
        x = x.permute(0, 3, 1, 2)

       
        x = self.upsample1(x)

        x_up = x
        x = self.output(x)
        return x, x_up

    def forward(self, x):
        x, kd_feature_encoder = self.forward_features(x)

        x, kd_feature_decoder = self.forward_up_features(x)

        x, x_up = self.up_x4(x)


        if self.training:
            return x,kd_feature_encoder,kd_feature_decoder,x_up
            print("feature",kd_feature_encoder)
        else:
            return x

