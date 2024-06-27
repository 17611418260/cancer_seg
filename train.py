import os, sys, math, torch
import torch.utils.data
from external.utils import *
from np_dataset import get_datasets
from model import get_mask_rcnn

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(3)
print('CUDA available:',torch.cuda.is_available())

# TODO Define train validate testing dataloaders
cancer_root_path = "/data/fengqianpang/DataBase/Kaggle/cancer-instance-segmentation"
trainset, valset, testset = get_datasets(root_path=cancer_root_path)

batch_size = 32  # run-96 debug-1
data_loader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn
)
data_loader_val = torch.utils.data.DataLoader(
    valset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn
)
# data_loader_test = torch.utils.data.DataLoader(
#     testset, batch_size=32, shuffle=False, num_workers=8, collate_fn=collate_fn
# )


# TODO Create Mask-RCNN Model
num_classes = 2
mask_rcnn = get_mask_rcnn(device=device, num_classes=num_classes)


# TODO Define Training Parameters
params = [p for p in mask_rcnn.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


# TODO Train the model

# Training one epoch
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    """
    train model for one epoch
    """
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        images = list(image.float() for image in images)  # 加了这句
        # print(
        #     'ID:', targets[0]['image_id'],
        #     'images:', images[0].shape, 'masks:', targets[0]['masks'].shape,
        #     'labels:', targets[0]['labels'].shape, 'boxes:', targets[0]['boxes'].shape,
        #     'label max:', torch.max(targets[0]['labels'])
        # )
        loss_dict = model(images, targets)
        # print('loss_dict:', loss_dict)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

def train_n_epoch(num_epochs, model, optimizer, data_loader, device):
    """
    train model for n epochs
    """
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
        # val set evaluate
        # test set evaluate
        # save model
        lr_scheduler.step()
    # save model



if __name__ == '__main__':
    # TODO Train the model
    num_epochs = 20
    train_n_epoch(num_epochs, mask_rcnn, optimizer, data_loader, device)