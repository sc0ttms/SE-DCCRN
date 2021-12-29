# README

![Platform](https://img.shields.io/badge/Platform-win10--64-lightgrey)
![Python](https://img.shields.io/badge/Python-3.7-orange)
![Pytorch](https://img.shields.io/badge/Pytoch-1.10.1-orange)

## 项目简介

语音增强DCCRN论文复现

## 重要特性

- 模型
  - [ ] DCCRN-C
  - [ ] DCCRN-C(light) 

- 数据集
  - [x] [dns_interspeech_2021_mandarin_100h](https://aistudio.baidu.com/aistudio/datasetdetail/119056)

## 相关参考链接

### 复现论文

- [DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement](https://arxiv.org/abs/2008.00264)

### 模型压缩

- 知识蒸馏
  - [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
  - [Learning Efficient Object Detection Models with Knowledge Distillation](http://cseweb.ucsd.edu/~mkchandraker/pdf/nips17_distillationdetection.pdf)
- 剪枝
  - [Pytorch prune tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html?highlight=prune)
- 量化
  - [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295)
  - [Pytorch quantization tutorial](https://pytorch.org/docs/stable/quantization.html)

### 项目

- [FullSubNet](https://github.com/haoxiangsnr/FullSubNet)
- [DeepComplexCRN](https://github.com/huyanxin/DeepComplexCRN)

## License

[![License](https://img.shields.io/badge/License-BSD-green)](./LICENSE)
