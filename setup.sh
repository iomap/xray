cd /home/iomap/kaggle


## Create Directory Tree ##

mkdir input
mkdir src
mkdir models
mkdir notebooks
touch README.md
touch LICENSE
mkdir input/images
mkdir datasets

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


