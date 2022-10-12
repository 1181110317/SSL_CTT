# SSL_CTT

This repository contains the official implementation of [Semi-Supervised Semantic Segmentation with Cross Teacher Training](https://www.sciencedirect.com/science/article/abs/pii/S0925231222010384),  [arxiv](https://arxiv.org/pdf/2209.01327.pdf).

This code is based on [ClassMix code](https://github.com/WilhelmT/ClassMix)



> **Abstract.** The crux of semi-supervised semantic segmentation is to assign adequate pseudo-labels to the pixels of unlabeled images. A common practice is to select the highly confident predictions as the pseudo ground-truth, but it leads to a problem that most pixels may be left unused due to their unreliability. We argue that every pixel matters to the model training, even its prediction is ambiguous. Intuitively, an unreliable prediction may get confused among the top classes (*i.e*., those with the highest probabilities), however, it should be confident about the pixel not belonging to the remaining classes. Hence, such a pixel can be convincingly treated as a negative sample to those most unlikely categories. Based on this insight, we develop an effective pipeline to make sufficient use of unlabeled data. Concretely, we separate reliable and unreliable pixels via the entropy of predictions, push each unreliable pixel to a category-wise queue that consists of negative samples, and manage to train the model with all candidate pixels. Considering the training evolution, where the prediction becomes more and more accurate, we adaptively adjust the threshold for the reliable-unreliable partition. Experimental results on various benchmarks and training settings demonstrate the superiority of our approach over the state-of-the-art alternatives.

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
mkdir ../data/VOC2012/
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
  author={Xiao, Hui and Li, Dong and Xu, Hao and Fu, Shuibo and Yan, Diqun and Song, Kangkang and Peng, Chengbin},
  journal={Neurocomputing},
  volume={508},
  pages={36--46},
  year={2022},
  publisher={Elsevier}
}
```

