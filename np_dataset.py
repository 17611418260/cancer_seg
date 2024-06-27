import os
import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import external.transforms as T
import external.utils


class CancerDataset(torch.utils.data.Dataset):

    def __init__(self, images, masks, valid_ids, transforms=None):  # 定义的全局变量必须加self.
        self.images, self.masks = images, masks
        self.valid_ids = valid_ids
        self.transforms = transforms  # transform用于数据增强

    def __getitem__(self, idx):
        idx = self.valid_ids[idx]
        img = (self.images[idx,:,:,:]-128)/256
        mask = self.masks[idx, :, :, 0]
        # img = np.transpose(img,(2,0,1))
        # print('masks shape:', img.shape)


        obj_ids = np.unique(mask)
        # print('obj ids:', obj_ids)
        # 去除切片维度，也就是[0]
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]  # 把每张图都拿出来放到masks里,拓展维度
        # print('masks shape:', masks.shape)

        # 给每个mask加框
        num_objs = len(obj_ids)
        boxes = []
        fliter_masks = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])  # 列是1，行是0
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if (xmax - xmin) > 3 and (ymax - ymin) > 3:
                # TODO boxes (``FloatTensor[N, 4]``): the predicted boxes
                #  in ``[x1, y1, x2, y2]`` format
                boxes.append([xmin, ymin, xmax, ymax])  # append：往列表里加元素
                # boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
                fliter_masks.append(masks[i])

        labels = torch.ones((len(boxes),), dtype=torch.int64) #- 1
        boxes = torch.as_tensor(boxes)
        fliter_masks = torch.as_tensor(fliter_masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        # image_id = torch.tensor([len(boxes)])
        # print('{}-th image has {} cancer cells'.format(idx,len(boxes)))

        # if idx in [34, 44, 99, 564, 643, 822]:
        #     plt.imshow(self.images[idx, :, :, :].astype(np.uint8))
        #     plt.title('image id: {}'.format(idx))
        #     plt.show()

        target = {}
        target["boxes"] = boxes
        target["masks"] = fliter_masks
        target["image_id"] = image_id
        target["labels"] = labels

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return self.valid_ids.shape[0]


def get_transform(train):
    transforms = []
    # 把PIL图转为张量
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))  # 随机水平翻转,翻一倍
    return T.Compose(transforms)


def filter_masks(masks):
    valid_idxs = []
    for idx in range(masks.shape[0]):
        m = masks[idx,:,:,0]
        obj_ids = np.unique(m)
        if len(obj_ids) == 1:
            continue
        else:
            obj_ids = obj_ids[1:]
            ms = m == obj_ids[:, None, None]  # 把每张图都拿出来放到masks里,拓展维度

            # 给每个mask加框
            num_objs = len(obj_ids)
            boxes = []
            for i in range(num_objs):
                pos = np.where(ms[i])
                xmin = np.min(pos[1])   # 列是1，行是0
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                # boxes.append([xmin, ymin, xmax-xmin, ymax-ymin])
                if (xmax-xmin) > 3 and (ymax-ymin) > 3:
                    boxes.append([xmin, ymin, xmax, ymax])
            if len(boxes) > 0:
                valid_idxs.append(idx)
    return np.array(valid_idxs)


def get_datasets(root_path,split=1):
    """
    root_path = "/data/fengqianpang/DataBase/Kaggle/cancer-instance-segmentation"
    """
    img_path = '{}/split{}/Images/images.npy'.format(root_path, split)
    images = np.load(img_path, mmap_mode='r+')
    mask_path = '{}/split{}/Masks/masks.npy'.format(root_path, split)
    masks = np.load(mask_path, mmap_mode='r+')

    valid_idxs = filter_masks(masks)
    # int(len(valid_idxs) * 0.8)  # 1459
    # 计算有效图片在原序列中的位置
    trn_id, val_id = int(len(valid_idxs) * 0.8), int(len(valid_idxs) * 0.9)
    print('filter ids:', trn_id, val_id)
    print('org ids:', valid_idxs[trn_id], valid_idxs[val_id])
    print('trainset len:', valid_idxs[:trn_id].shape)
    print('valset len:', valid_idxs[trn_id:val_id].shape)
    print('testset len:', valid_idxs[val_id:].shape)

    images_split = np.split(images,[valid_idxs[trn_id],valid_idxs[val_id]])
    print('images_split 0 shape:', images_split[0].shape)

    masks_split = np.split(masks, [valid_idxs[trn_id], valid_idxs[val_id]])
    print('masks_split 1 shape:', masks_split[1].shape)

    tr_idxs = valid_idxs[:trn_id]
    val_idxs = valid_idxs[trn_id:val_id] - valid_idxs[trn_id]
    ts_idxs = valid_idxs[val_id:] - valid_idxs[val_id]

    trainset = CancerDataset(images_split[0], masks_split[0], tr_idxs, get_transform(train=True))
    valset = CancerDataset(images_split[1], masks_split[1], val_idxs, get_transform(train=False))
    testset = CancerDataset(images_split[2], masks_split[2], ts_idxs, get_transform(train=False))
    print('trainset len:', len(trainset), 'valset len:', len(valset), 'testset len:', len(testset))

    return trainset, valset, testset














