import os
import torch
import json


path = '/data1/lvis_test1/'
output_path = '/data1/lvis_test1/all.dist'


output_dict = {}
for file in os.listdir(path):
    if file.endswith('.dist'):
        name = file.split('.')[0]
        output_dict[name] = torch.load(path + file)


torch.save(output_dict, output_path)