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
