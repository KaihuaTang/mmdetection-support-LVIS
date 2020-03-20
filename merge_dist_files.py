import os
import torch
import json
from .mmdet.models.detectors.htc import SAVE_PATH, ALL_DIST_PATH

path = SAVE_PATH
output_path = ALL_DIST_PATH


output_dict = {}
for file in os.listdir(path):
    if file.endswith('.dist'):
        name = file.split('.')[0]
        output_dict[name] = torch.load(path + file)


torch.save(output_dict, output_path)