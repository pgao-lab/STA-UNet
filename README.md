# STA-UNet
The codes for the work "Transformer UNet with Super Token Attention for Medical Image Segmentation".


## 1. Prepare data

The datasets we used are provided by TransUnet's authors. [Get processed data in this link] (https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd).

## 2. Train/Test

- Run the train script on synapse dataset. The batch size we used is 24.If you do not have enough GPU memory, the bacth size can be reduced to 12 or 6 to save memory.

- Train

```bash
sh train.sh 
```

- Test 

```bash
sh test.sh 
```
##Pretrained Weights
The pretrained checkpoint stvit_small_224.pth can be downloaded from [ https://drive.google.com/file/d/10s79is5YGr4BpBJNTy15ak9Mk3tA9PSm/view?usp=drive_link].

After downloading, place it under:staunet/pretrained_ckpt/stvit_small_224.pth
## References
* [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
* [STViT](https://github.com/hhb072/STViT)
##Citation
@article{gao2026transformer,
  title={Transformer UNet with super token attention for medical image segmentation},
  author={Gao, Peng and Xia, Ling-Xin and Liu, Xiao and Wang, Fei and Yuan, Ru-Yue},
  journal={Applied Soft Computing},
  pages={115752},
  year={2026},
  publisher={Elsevier}
}

