# STA-UNet
The codes for the work "Transformer UNet with Super Token Attention for Medical Image Segmentation".

## 1. Download pre-trained swin transformer model (Swin-T)
* [Get pre-trained model in this link] (https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing): Put pretrained Swin-T into folder "pretrained_ckpt/"

## 2. Prepare data

The datasets we used are provided by TransUnet's authors. [Get processed data in this link] (https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd).

## 3. Train/Test

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


