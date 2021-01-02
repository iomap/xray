# this script should be called in 'xray' directory

## Create Directory Tree ##

mkdir input
mkdir src
mkdir models
mkdir notebooks
mkdir input/images
mkdir datasets

## Install Dependencies ##

#cd src
#git clone https://github.com/ultralytics/yolov5  # clone repo
#cd yolov5
#pip install -qr requirements.txt pycocotools # install dependencies
#cd ../..

## Download Datasets ##

cd datasets
kaggle competitions download -c vinbigdata-chest-xray-abnormalities-detection
kaggle datasets download awsaf49/vinbigdata-yolo-labels-dataset
kaggle datasets download awsaf49/vinbigdata-1024-image-dataset

## Unzip Datasets ##

mkdir vinbigdata-chest-xray-abnormalities-detection
cd vinbigdata-chest-xray-abnormalities-detection
unzip ../vinbigdata-chest-xray-abnormalities-detection.zip

mkdir vinbigdata-yolo-labels-dataset
cd vinbigdata-yolo-labels-dataset
unzip ../vinbigdata-yolo-labels-dataset.zip

mkdir vinbigdata-1024-image-dataset
cd vinbigdata-1024-image-dataset
unzip ../vinbigdata-1024-image-dataset.zip


