# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging

import torch
import torch.nn as nn

from .stvit_unet import STViTUnetsys

logger = logging.getLogger(__name__)


class STViTUnet(nn.Module):
    def __init__(self, config, num_classes=21843, zero_head=False, vis=False):
        super(STViTUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.stvit_unet = STViTUnetsys(
            in_chans=config.MODEL.STViT.IN_CHANS,
            num_classes=self.num_classes,
            embed_dim=config.MODEL.STViT.EMBED_DIM,
            depth=config.MODEL.STViT.DEPTH,
            num_heads=config.MODEL.STViT.NUM_HEADS,
            seg_layers=config.MODEL.STViT.SEG_LAYERS,
            stoken_size=config.MODEL.STViT.STOKEN_SIZE,
            projection=config.MODEL.STViT.PROJECTION,
            mlp_ratio=config.MODEL.STViT.MLP_RATIO,
            qkv_bias=config.MODEL.STViT.QKV_BIAS,
            qk_scale=config.MODEL.STViT.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE)
        torch.save(self.stvit_unet.state_dict(), 'stvit_unet0.pth')
        print("STViTUnet model is saved.")

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.stvit_unet(x)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        # print('pretrained_path')
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            pretrained_dict = pretrained_dict['model'] 


            model_dict = self.stvit_unet.state_dict()  


            new_dict = {}
            for k, v in pretrained_dict.items():
                if "layers" in k and "blocks" in k:
                    
                    layer_num = int(k.split('.')[1])  # 例如 'layers.0.blocks'，则 layer_num 为 0

                    
                    new_stage_name = f"stage{layer_num + 1}"  # stage1, stage2, stage3, stage4

                    
                    new_k = k.replace(f"layers.{layer_num}.blocks", new_stage_name)

                    
                    new_dict[new_k] = v

                    
                    stage_up_k = new_k.replace(f"stage{layer_num + 1}", f"stage_up{layer_num + 1}")

                    new_dict[stage_up_k] = v
                else:
                    
                    new_dict[k] = v

            for k in list(new_dict.keys()):
                if k in model_dict:
                    if new_dict[k].shape != model_dict[k].shape:
                        print(
                            "delete:{};shape pretrain:{};shape model:{}".format(k, new_dict[k].shape,
                                                                                model_dict[k].shape))
                        del new_dict[k]

            msg = self.stvit_unet.load_state_dict(new_dict, strict=False)
            print("Loaded state dict:", msg)

            loaded_keys = set(new_dict.keys()) - set(msg.missing_keys) - set(msg.unexpected_keys)
            print("Successfully loaded keys:", loaded_keys)

        else:
            print("none pretrain")



