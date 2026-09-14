import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import torch
import sys
import argparse
import os


yaml_path = r'models/SBP-YOLO.yaml'
data_path = r'datasets/yolo_0422-ALL_7K5_7_1_1.yaml'

def main(opt):
    yaml = opt.cfg
    data = opt.data
    model = YOLO(yaml)
    model.info()
    results = model.train(
                cache = False,
                data =  data,
                imgsz=640,  # training image size, default 640
                epochs=300,  # training epochs, default 100
                batch=16,  # training batch size, default options 16, 32, 64
                project='zzz',
                name='exp',  # folder name for saving training runs, default 'exp', increments sequentially
                #device='0',  # device to run on: '0' for GPU training, 'cpu' for CPU
                patience=30,
                seed=0,
                lr0=0.001,
                optimizer='Adam', # Adam 'SGD', 'AdamW', 'NAdam', 'RAdam'
                close_mosaic=0,
                amp=True,
                # amp=False,
                workers=8, # needs to be set to 8
                simplify=True # default is true
            )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=yaml_path, help='initial weights path')
    parser.add_argument('--weights', type=str, default='', help='')
    parser.add_argument('--data', type=str, default=data_path, help='data  path')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
