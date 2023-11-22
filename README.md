# A-Follo
AIFFEL 강남 4기 해커톤 팀 아폴로입니다.
<br/><br/><br/>

## Project
위성 영상에서 항공기, 선박, 차량 Class에 대한 위치 식별(Object Detection)
<br/><br/><br/>

## Problem Definition

### FP
잘못된 객체 오인식

### FN
밀집된 객체군에서 Detection 누락
<br/>주로 고해상도(큰 Size 이미지)이므로 Downsampling 과정에서 화질 저하 발생
<br/><br/><br/>

## Project

### Data
* Patch
* Size Based Posted Filtering by GSD
* Augmentation

### Model
* Backbone

### Further investigation
* Semantic Segmentation
* Hyperparameter Tuning
* Transfer Learning
<br/><br/><br/>

## Library
`MMRotate` `GDAL` `Shapely` `Albumentation`
<br/><br/>

## Stack
`Python` `QGIS`
<br/><br/>

## Development Envrionment
* Data Storage : Google Cloud
* Code Editor : Google Colab
* GPU : K80
* RAM : 52GB
<br/><br/>

## Reference
Xian Sun et al. FAIR1M: A Benchmark Dataset for Fine-grained Object Recognition in High-Resolution Remote Sensing Imagery. ISPRS, 2022.
<br/>Yue Zhou et al. MMRotate: A Rotated Object Detection Benchmark using PyTorch. 2022,  ACM MM.
<br/>Kai Chen et al. MMDetection: Open MMLab Detection Toolbox and Benchmark. 2019, arXiv preprint arXiv:1906.07155.
<br/>Jian Ding et al. Learning RoI Transformer for Oriented Object Detection in Aerial Images. 2019, IEEE/CVF.
<br/>Kaiming He et al. Deep Residual Learning for Image Recognition. 2016, IEEE.
<br/>Gui-Song Xia et al. DOTA: A Large-scale Dataset for Object Detection in Aerial Images. 2018, CVPR.
<br/>Alexander Buslaev et al. Albumentations: fast and flexible image augmentations. 2020, Information.
<br/>Ross Girshick et al. Rich feature hierarchies for accurate object detection and semantic segmentation. 2014, CVPR.
<br/>Jan Hosang et al. Learning non-maximum suppression. 2017, CVPR.
<br/>Hamid Rezatofighi et al. Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression. 2019, CVPR.
<br/><br/><br/>
