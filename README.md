# GSAN
# Graph-based Structural Attributes for Vehicle Re-identification
This repo gives the code for the paper "Rongbo Zhang, Xian Zhong, Xiao Wang, Wenxin Huang, Wenxuan Liu, Shilei Zhao: Graph-based Structural Attributes for Vehicle Re-identification.ICME 2022.
This code is based on [Beyond the Parts: 
Learning Multi-view Cross-part Correlation for Vehicle Re-identification](https://lxc86739795.github.io/papers/2020_ACMMM_PCRNet.pdf). ACM MM 2020".

## Requirements

- Linux or macOS with python ≥ 3.6
- PyTorch ≥ 1.0
- torchvision that matches the Pytorch installation. You can install them together at [pytorch.org](https://pytorch.org/) to make sure of this.
- [yacs](https://github.com/rbgirshick/yacs)
- Cython (optional to compile evaluation code)
- tensorboard (needed for visualization): `pip install tensorboard`

## Data Preparation

- 1.You need the original image datasets like [VeRi](https://github.com/JDAI-CV/VeRidataset) and the parsing masks of all images.
For a vehicle parsing model pretrained on the [MVP dataset](https://lxc86739795.github.io/MVP.html) based on PSPNet or HRNet, please contact [Xinchen Liu](https://lxc86739795.github.io/).
- 2.You need the original image datasets like [VeRi](https://github.com/JDAI-CV/VeRidataset) and the attribute masks of all images.
For a vehicle parsing model pretrained on the [MVP dataset](https://lxc86739795.github.io/MVP.html) based on YOLOv4, please contact [Rongbo Zhang](https://HappyBoBo0331.github.io/).
## Training

You can run the examplar training script in `.sh` files.

## Main Code

The main code for GCN can be found in 
```bash
root
  engine
    trainer_selfgcn.py    # training pipline
  modeling
    baseline_selfgcn.py   # definition of the model
  tools
    train_selfgcn.py      # training preparation

```

Here https://pan.baidu.com/s/1N1EBv6i58lOm4EbXNbio5w (Extraction code: rysd). This VAT dataset is for research purposes only and therefore not for commercial use.

## Reference
```BibTeX

```
