# SOURCES
#https://www.kaggle.com/awsaf49/vinbigdata-cxr-ad-yolov5-14-class
#https://www.kaggle.com/ultralytics/yolov5

# install
#git clone https://github.com/ultralytics/yolov5  # clone repo
#cd yolov5
#pip install -qr requirements.txt pycocotools # install dependencies

import torch
from IPython.display import Image, clear_output  # to display images

clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))


