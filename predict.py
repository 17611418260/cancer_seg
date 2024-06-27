from __future__ import print_function
from collections import defaultdict, deque
import datetime
import pickle
import time
import torch.distributed as dist
import errno
import math
import collections
import os
import numpy as np
import torch
import torch.utils.data
from skimage.measure import label
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision
import random
import sys
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
ImageFile.LOAD_TRUNCATED_IMAGES = True



from external.utils import *









import os
import numpy as np

root = 'C:/Users/Administrator/Desktop'
img_path = os.path.join(root, 'images_test.npy')
images = np.load(img_path)
mask_path = os.path.join(root, 'masks_test.npy')
masks = np.load(mask_path)

testset = CancerDataset(images, masks, get_transform(train=True))



data_loader_test = torch.utils.data.DataLoader(
    testset, batch_size=2, shuffle=True, num_workers=0,
    collate_fn=lambda x: tuple(zip(*x)))

# create mask rcnn model
num_classes = 1
device = torch.device('cpu')
model_ft = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
print('model_ft: ', model_ft)
in_features = model_ft.roi_heads.box_predictor.cls_score.in_features
print('in_features: ', in_features)
model_ft.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
in_features_mask = model_ft.roi_heads.mask_predictor.conv5_mask.in_channels
print('in_features_mask: ', in_features_mask)
hidden_layer = 256
model_ft.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
model_ft.to(device)

for param in model_ft.parameters():
    param.requires_grad = True




def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
        # 加了utils

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        images = list(image.float() for image in images)  # 加了这句

        loss_dict = model_ft(images, targets)
        print('loss_dict:', loss_dict)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)  # 加了utils
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()  # 加了这句

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])



params = [p for p in model_ft.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=5,
                                               gamma=0.1)



num_epochs = 20
for epoch in range(num_epochs):
    train_one_epoch(model_ft, optimizer, data_loader_test, device, epoch, print_freq=100)
    lr_scheduler.step()

