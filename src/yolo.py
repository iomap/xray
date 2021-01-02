import config

import torch
from IPython.display import Image, clear_output  # to display images
import os
from os import listdir
from os.path import isfile, join
import yaml
from glob import glob
import numpy as np, pandas as pd
from glob import glob
import shutil, os
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from tqdm.notebook import tqdm
import seaborn as sns
os.chdir('yolov5')

def train(yaml_path='vinbigdata.yaml'):
    os.system(f'WANDB_MODE="dryrun" python train.py --img 512 --batch 16 --epochs 5 --data {yaml_path} --weights {config.YOLO_VERSION}.pt --cache')

def detect(img_path):
    os.system(f'python detect.py --weights {config.YOLO_VERSION}.pt --img 640 --conf 0.25 --source {img_path}')
    #Image(filename='runs/detect/exp/zidane.jpg', width=600)

def torch_version():
    print('Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

def create_train_yaml():
    cwd = os.getcwd()+'/'
    print('PC_DEBUG: ', cwd)
    dataset_path = '/home/iomap/xray/input/vinbigdata-1024-image-dataset'
    
    with open(join( cwd , 'train.txt'), 'w') as f:
        for path in glob(f'{dataset_path}/vinbigdata/train/*'):
            f.write(path+'\n')
                
    with open(join( cwd , 'val.txt'), 'w') as f:
        for path in glob(f'{dataset_path}/vinbigdata/val/*'):
            f.write(path+'\n')
    
    data = dict(
        train =  join( cwd , 'train.txt') ,
        val   =  join( cwd , 'val.txt' ),
        nc    = 14,
        names = classes
        )
    
    with open(join( cwd , 'vinbigdata.yaml'), 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    
    f = open(join( cwd , 'vinbigdata.yaml'), 'r')
    print('\nyaml:')
    print(f.read())


# MAIN #
detect('/home/iomap/xray/datasets/vinbigdata-1024-image-dataset/vinbigdata/train/a9f17d3eb4a8221c3ef0dd12bfec0ba0.png')

train_df = pd.read_csv('/home/iomap/xray/datasets/vinbigdata-1024-image-dataset/vinbigdata/train.csv')
train_df.head()
train_df['image_path'] = '/home/iomap/xray/datasets/vinbigdata-1024-image-dataset/vinbigdata/train/'+train_df.image_id+'.png'
train_df.head()
train_df = train_df[train_df.class_id!=14].reset_index(drop = True)

gkf  = GroupKFold(n_splits = 5)
train_df['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(gkf.split(train_df, groups = train_df.image_id.tolist())):
    train_df.loc[val_idx, 'fold'] = fold
train_df.head()

train_files = []
val_files   = []
val_files += list(train_df[train_df.fold==0].image_path.unique())
train_files += list(train_df[train_df.fold!=0].image_path.unique())
len(train_files), len(val_files)


os.makedirs('/home/iomap/xray/input/vinbigdata/labels/train', exist_ok = True)
os.makedirs('/home/iomap/xray/input/vinbigdata/labels/val', exist_ok = True)
os.makedirs('/home/iomap/xray/input/vinbigdata/images/train', exist_ok = True)
os.makedirs('/home/iomap/xray/input/vinbigdata/images/val', exist_ok = True)
label_dir = '/home/iomap/xray/datasets/vinbigdata-yolo-labels-dataset/labels'
for file in tqdm(train_files):
    shutil.copy(file, '/home/iomap/xray/input/vinbigdata/images/train')
    filename = file.split('/')[-1].split('.')[0]
    shutil.copy(os.path.join(label_dir, filename+'.txt'), '/home/iomap/xray/input/vinbigdata/labels/train')

for file in tqdm(val_files):
    shutil.copy(file, '/home/iomap/xray/input/vinbigdata/images/val')
    filename = file.split('/')[-1].split('.')[0]
    shutil.copy(os.path.join(label_dir, filename+'.txt'), '/home/iomap/xray/input/vinbigdata/labels/val')

class_ids, class_names = list(zip(*set(zip(train_df.class_id, train_df.class_name))))
classes = list(np.array(class_names)[np.argsort(class_ids)])
classes = list(map(lambda x: str(x), classes))
classes

create_train_yaml()

train('/home/iomap/xray/src/yolov5/vinbigdata.yaml')
