import os
import glob
import argparse
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image
import torch

import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset

results = mmcv.load("/data/songziying/workspace/FusionAD/UniAD/output/results.pkl")
sub_results = []
for i in range(6019):
    sub_dict = {}
    sub_dict['plan_results'] = results['bbox_results'][i]['planning_traj']
    sub_results.append(sub_dict)
mmcv.dump(sub_results, '/data/songziying/workspace/FusionAD/UniAD/output/my.pkl')
#import pdb;pdb.set_trace(sub_dict)
#print("hhhhhhhhhh")