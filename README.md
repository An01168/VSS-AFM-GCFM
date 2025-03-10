# VSS-AFM-GCFM
Improving Video Semantic Segmentation via Adaptive and Global Temporal Contexts Modeling

This repository is the official implementation of "Improving Video Semantic Segmentation via Adaptive and Global Temporal Contexts Modeling” ( This paper is under submission, we will show it later)

## Install & Requirements
Requirements: `PyTorch >= 1.4.0, CUDA >= 10.0, and Python==3.8`

**To Install weightingFunction**
```
cd $VSS-AFM-GCFM_ROOT/Local-Attention-master
python setup.py build
```
**To Install Correlation**
```
cd $VSS-AFM-GCFM_ROOT/correlation
python setup.py build
```
## Usage
### Data preparation
Please follow [Cityscapes](https://www.cityscapes-dataset.com/) to download Cityscapes dataset, or [Camvid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) to download Camvid dataset. After correctly downloading, the file system is as follows:
````bash
$VSS-AFM-GCFM_ROOT/VSS-Dataset-BasicModel/data
├── Cityscapes_video
│   ├── gtFine
│   │   ├── train
│   │   └── val
│   └── leftImg8bit_sequence
│       ├── train
│       └── val
````
````bash
$VSS-AFM-GCFM_ROOT/VSS-Dataset-BasicModel/data
├── Camvid_video
│   ├── train
│   ├── test
│   ├── trainannot
│   ├── testannot

````
### Training

1. cd $VSS-AFM-GCFM_ROOT/VSS-Dataset-BasicModel

2. Download pretrained models [BaiduYun(Access Code:zjeu)]( https://pan.baidu.com/s/15lJ2-iMADEoPuqN1_H6Xyw), and put them in a folder `./ckpt`.

3. Training requires 4 Nvidia GPUs.
````bash
bash ./train.sh
````
### Test
1. Download the trained weights from [BaiduYun(Access Code:5smj)]( https://pan.baidu.com/s/1anQNL-tCMjKu2tBHCw34ZQ) and put them in a folder `./ckpt`.

2. Run the following commands:
````bash
bash ./eval_multipro.sh
````
## Acknowledgement
The code is heavily based on the following repositories:
- https://github.com/CSAILVision/sceneparsing
- https://github.com/zzd1992/Image-Local-Attention
- https://github.com/WeilunWang/correlation-layer

Thanks for their amazing works.

## Citation
We will show it later.



