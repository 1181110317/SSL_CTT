# SSL_CTT

This repository contains the official implementation of the following manuscript: 
Hui Xiao, Li Dong, Hao Xu, Shuibo Fu, Diqun Yan, Kangkang Song, and Chengbin Peng. "Semi-supervised semantic segmentation with cross teacher training." Neurocomputing 508 (2022): 36-46.  
[Neurocomputing](https://www.sciencedirect.com/science/article/abs/pii/S0925231222010384),  [arxiv](https://arxiv.org/pdf/2209.01327.pdf).

This code is based on [ClassMix code](https://github.com/WilhelmT/ClassMix)



> **Abstract.** Convolutional neural networks can achieve remarkable performance in semantic segmentation tasks. However, such neural network approaches heavily rely on costly pixel-level annotation. Semi-supervised learning is a promising resolution to tackle this issue, but its performance still far falls behind the fully supervised counterpart. This work proposes a cross-teacher training framework with three modules that significantly improves traditional semi-supervised learning approaches. The core is a cross-teacher module, which could simultaneously **reduce the coupling among peer networks and the error accumulation between teacher and student networks**. In addition, we propose two complementary contrastive learning modules. The high-level module can **transfer high-quality knowledge from labeled data to unlabeled ones and promote separation between classes in feature space**. The low-level module can **encourage low-quality features learning from the high-quality features among peer networks**. In experiments, the cross-teacher module significantly improves the performance of traditional student–teacher approaches, and our framework outperforms state-of-the-art methods on benchmark datasets.

[![img](https://github.com/1181110317/SSL_CTT/blob/main/img/pipeline.jpg)](https://github.com/1181110317/SSL_CTT/blob/main/img/pipeline.jpg)

## Prerequisites

- CUDA/CUDNN
- Python3
- Packages found in requirements.txt

## Datasets

```
mkdir ../dataset/CityScapes/
```

Download the dataset from [here](https://www.cityscapes-dataset.com/).

### Pascal VOC 2012

```
mkdir ../dataset/VOC2012/
```

Download the dataset from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).

## Experiments

#### For example, for Cityscapes with deeplabv2:

```
python3 cross_teacher_contr.py --config ./configs/configCityscapes.json --name deeplabv2_city
```

#### For example, for PASCAL VOC with deeplabv2:

```
python3 cross_teacher_contr.py --config ./configs/configVOC.json --name deeplabv2_voc
```

#### For example, for PASCAL VOC with deeplabv3+:

```
python3 cross_teacher_deeplabv3+.py --config ./configs/configCityscapes.json --name voc
```

## Citation

```
@article{xiao2022semi,
  title={Semi-supervised semantic segmentation with cross teacher training},
  author={Xiao, Hui and Dong, Li and Xu, Hao and Fu, Shuibo and Yan, Diqun and Song, Kangkang and Peng, Chengbin},
  journal={Neurocomputing},
  volume={508},
  pages={36--46},
  year={2022},
  publisher={Elsevier}
}
```

