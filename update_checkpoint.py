import json
import torch

OLD_DIM = 271
PAD_DIM = 320
old_path = './work_dirs/270_x64/epoch_20_no_bias.pth'
new_path = './work_dirs/270_x64/epoch_20_pad.pth'
old_checkpoint = torch.load(old_path)

def check_dimension(checkpoint):
    for key, val in old_checkpoint['state_dict'].items():
        if (len(val.shape) >= 1) and (val.shape[0] == OLD_DIM):
            print(key, ' : ', val.shape)

def update_state_dict(checkpoint):
    for key, val in old_checkpoint['state_dict'].items():
        if (len(val.shape) >= 1) and (val.shape[0] == OLD_DIM):
            pad_shape = list(val.shape)
            pad_shape[0] = PAD_DIM
            pad_tensor = torch.zeros(pad_shape).normal_(mean=0, std=0.01).to(val.device)
            old_checkpoint['state_dict'][key] = torch.cat([val, pad_tensor], dim=0)

def check_new_dimension(checkpoint):
    for key, val in old_checkpoint['state_dict'].items():
        if (len(val.shape) >= 1) and (val.shape[0] == OLD_DIM+PAD_DIM):
            print('new: ', key, ' : ', val.shape)


check_dimension(old_checkpoint)
update_state_dict(old_checkpoint)
check_new_dimension(old_checkpoint)


def check_optimizer_dimension(checkpoint):
    for key, val in old_checkpoint['optimizer']['state'].items():
        val = val['momentum_buffer']
        if (len(val.shape) >= 1) and (val.shape[0] == OLD_DIM):
            print(key, ' : ', val.shape)

def update_optimizer_state(checkpoint):
    old_checkpoint['optimizer']['param_groups'][0]['initial_lr'] = 0.005
    for key, val in old_checkpoint['optimizer']['state'].items():
        val = val['momentum_buffer']
        if (len(val.shape) >= 1) and (val.shape[0] == OLD_DIM):
            pad_shape = list(val.shape)
            pad_shape[0] = PAD_DIM
            pad_tensor = torch.zeros(pad_shape).normal_(mean=0, std=0.01).to(val.device)
            old_checkpoint['optimizer']['state'][key]['momentum_buffer'] = torch.cat([val, pad_tensor], dim=0)

def check_optimizer_new_dimension(checkpoint):
    for key, val in old_checkpoint['optimizer']['state'].items():
        val = val['momentum_buffer']
        if (len(val.shape) >= 1) and (val.shape[0] == OLD_DIM + PAD_DIM):
            print('new: ', key, ' : ', val.shape)

#check_optimizer_dimension(old_checkpoint)
#update_optimizer_state(old_checkpoint)
#check_optimizer_new_dimension(old_checkpoint)

old_checkpoint.pop('optimizer')

torch.save(old_checkpoint, new_path)