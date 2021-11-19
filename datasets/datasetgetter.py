"""
TUNIT: Truly Unsupervised Image-to-Image Translation
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import torch
from torchvision.datasets import ImageFolder
import os
import torchvision.transforms as transforms
from datasets.custom_dataset import ImageFolerRemap, CrossdomainFolder
from PIL import Image
import random
import numpy as np
from torch.autograd import Variable

"""
def sobel_process(imgs, include_rgb=True, using_IR=False):
    #print(imgs)
    c,h, w = imgs.size()

    if not using_IR:
        grey_imgs = imgs[ 2, :, :].unsqueeze(1)
        rgb_imgs = imgs[ :2, :, :]

    sobel1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv1.weight = torch.nn.Parameter(
        torch.Tensor(sobel1).float().unsqueeze(0).unsqueeze(0))
    dx = conv1(Variable(grey_imgs)).data

    sobel2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv2.weight = torch.nn.Parameter(
        torch.from_numpy(sobel2).float().unsqueeze(0).unsqueeze(0))
    dy = conv2(Variable(grey_imgs)).data

    sobel_imgs = torch.cat([dx, dy], dim=1)

    sobel_imgs = torch.cat([rgb_imgs, sobel_imgs], dim=1)


    return sobel_imgs
"""


class DuplicatedCompose(object):
    def __init__(self, tf1, tf2,tf3):
        self.tf1 = tf1
        self.tf2 = tf2
        self.tf3 = tf3

    def __call__(self, img):
        img1 = img.copy()
        img2 = img.copy()
        img3 = img.copy()
        for t1 in self.tf1:
            img1 = t1(img1)
        #img1 = sobel_process(img1)
        for t2 in self.tf2:
            img2 = t2(img2)
        #img2 = sobel_process(img2)
        for t3 in self.tf3:
            img3 = t3(img3)
        #img3 = sobel_process(img3)
        return img1, img2,img3


def get_dataset(args):

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    normalize = transforms.Normalize(mean=mean, std=std)
    #                                       transforms.RandomInvert(),transforms.RandomEqualize(p=0.5),
    if args.augmentation == "mnist":
        transform = DuplicatedCompose([transforms.Resize((args.img_size, args.img_size)),
                                       transforms.ToTensor(),
                                       normalize],
                                      [transforms.Pad(int(args.img_size * random.randint(1, 4) * 0.1)),
                                      transforms.RandomRotation(25, resample=Image.BILINEAR),
                                      transforms.Resize((args.img_size, args.img_size)),
                                      transforms.RandomChoice([
                                            transforms.RandomCrop(int(args.img_size * random.randint(7, 9) *  0.1)),
                                            transforms.CenterCrop(int(args.img_size * random.randint(7, 9) *  0.1))
                                          ]),
                                      transforms.Resize((args.img_size, args.img_size)),
                                      transforms.RandomInvert(),
                                      transforms.ColorJitter(0.4, 0.4, 0.4, 0.125),
                                       transforms.ToTensor(), normalize],
                                      [transforms.Resize((args.img_size, args.img_size)),
                                        transforms.RandomCrop(int(args.img_size * random.randint(5, 9) *  0.1)),
                                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.125),
                                        transforms.Resize(args.img_size),
                                        transforms.ToTensor(),
                                        normalize])


        transform_val = DuplicatedCompose([transforms.Resize((args.img_size, args.img_size)),
                                           transforms.ToTensor(),
                                           normalize],
                                          [transforms.RandomAffine(20, translate=(0.1, 0.1), shear=10),
                                           transforms.RandomResizedCrop(args.img_size, scale=(0.9, 1.1),
                                                                        ratio=(0.9, 1.1), interpolation=2),
                                           transforms.ToTensor(), normalize],
                                          [transforms.Resize((args.img_size, args.img_size)),
                                            transforms.CenterCrop(int(args.img_size * 0.5)),
                                            transforms.Resize(args.img_size),
                                            transforms.ToTensor(),
                                            normalize])

    elif args.augmentation == "original":
        transform = DuplicatedCompose([transforms.Resize((args.img_size, args.img_size)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalize],
                                      [transforms.RandomAffine(20, translate=(0.1, 0.1), shear=10),
                                       transforms.RandomResizedCrop(args.img_size, scale=(0.9, 1.1),
                                                                    ratio=(0.9, 1.1), interpolation=2),
                                       transforms.ColorJitter(0.4, 0.4, 0.4, 0.125),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(), normalize],
                                      [transforms.Resize((args.img_size, args.img_size)),
                                        transforms.CenterCrop(int(args.img_size * 0.5)),
                                        transforms.Resize(args.img_size),
                                        transforms.ToTensor(),
                                        normalize])


        transform_val = DuplicatedCompose([transforms.Resize((args.img_size, args.img_size)),
                                           transforms.ToTensor(),
                                           normalize],
                                          [transforms.RandomAffine(20, translate=(0.1, 0.1), shear=10),
                                           transforms.RandomResizedCrop(args.img_size, scale=(0.9, 1.1),
                                                                        ratio=(0.9, 1.1), interpolation=2),
                                           transforms.ToTensor(), normalize],
                                          [transforms.Resize((args.img_size, args.img_size)),
                                            transforms.CenterCrop(int(args.img_size * 0.5)),
                                            transforms.Resize(args.img_size),
                                            transforms.ToTensor(),
                                            normalize])

    elif args.augmentation == "improved_v0":
        transform = DuplicatedCompose([transforms.CenterCrop(args.img_size),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalize],
                                      [transforms.RandomAffine(20, translate=(0.1, 0.1), shear=10),
                                       transforms.RandomRotation(360, resample=Image.BILINEAR),
                                       transforms.CenterCrop(args.img_size),
                                       transforms.ColorJitter(0.4, 0.4, 0.4, 0.125),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(), normalize])

        transform_val = DuplicatedCompose([transforms.CenterCrop(args.img_size),
                                           transforms.ToTensor(),
                                           normalize],
                                          [transforms.RandomAffine(20, translate=(0.1, 0.1), shear=10),
                                           transforms.RandomRotation(360, resample=Image.BILINEAR),
                                           transforms.CenterCrop(args.img_size),
                                           transforms.ToTensor(), normalize])

    elif args.augmentation == "improved":
        transform = DuplicatedCompose([transforms.CenterCrop(args.img_size),
                                       transforms.ToTensor(),
                                       normalize],
                                      [transforms.RandomAffine(20, translate=(0.1, 0.1), shear=10),
                                       transforms.RandomRotation(360, resample=Image.BILINEAR),
                                       transforms.CenterCrop(args.img_size),
                                       transforms.ColorJitter(0.4, 0.4, 0.4, 0.125),
                                       transforms.ToTensor(), normalize])

        transform_val = DuplicatedCompose([transforms.CenterCrop(args.img_size),
                                           transforms.ToTensor(),
                                           normalize],
                                          [transforms.RandomAffine(20, translate=(0.1, 0.1), shear=10),
                                           transforms.RandomRotation(360, resample=Image.BILINEAR),
                                           transforms.CenterCrop(args.img_size),
                                           transforms.ToTensor(), normalize])
    elif args.augmentation == "improved_v2":
        transform = DuplicatedCompose([transforms.RandomRotation(360, resample=Image.BILINEAR),
                                       transforms.CenterCrop(args.img_size),
                                       transforms.ToTensor(),
                                       normalize],
                                      [transforms.RandomAffine(20, translate=(0.1, 0.1), shear=10),
                                       transforms.RandomRotation(360, resample=Image.BILINEAR),
                                       transforms.CenterCrop(args.img_size),
                                       transforms.ColorJitter(0.4, 0.4, 0.4, 0.125),
                                       transforms.ToTensor(), normalize])

        transform_val = DuplicatedCompose([transforms.CenterCrop(args.img_size),
                                           transforms.ToTensor(),
                                           normalize],
                                          [transforms.RandomAffine(20, translate=(0.1, 0.1), shear=10),
                                           transforms.RandomRotation(360, resample=Image.BILINEAR),
                                           transforms.CenterCrop(args.img_size),
                                           transforms.ToTensor(), normalize])
    elif args.augmentation == "BBC":
        transform = DuplicatedCompose([transforms.RandomRotation(360, resample=Image.BILINEAR),
                                       transforms.CenterCrop(args.img_size),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       normalize],
                                      [transforms.RandomAffine(20, translate=(0.1, 0.1), shear=10),
                                       transforms.RandomRotation(360, resample=Image.BILINEAR),
                                       transforms.CenterCrop(args.img_size),
                                       transforms.ColorJitter(0.4, 0.4, 0.4, 0.125),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(), normalize],
                                      [transforms.Resize((args.img_size, args.img_size)),
                                       transforms.RandomRotation(360, resample=Image.BILINEAR),
                                       transforms.RandomAffine(20, translate=(0.1, 0.1), shear=10),
                                       transforms.CenterCrop(int(args.img_size * 0.4)),
                                       transforms.Resize(args.img_size),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ColorJitter(0.4, 0.4, 0.4, 0.125),
                                       transforms.ToTensor(),
                                       normalize])

        transform_val = DuplicatedCompose([transforms.CenterCrop(args.img_size),
                                           transforms.ToTensor(),
                                           normalize],
                                          [transforms.RandomAffine(20, translate=(0.1, 0.1), shear=10),
                                           transforms.RandomRotation(360, resample=Image.BILINEAR),
                                           transforms.CenterCrop(args.img_size),
                                           transforms.ToTensor(), normalize],
                                          [transforms.Resize((args.img_size, args.img_size)),
                                           transforms.RandomRotation(360, resample=Image.BILINEAR),
                                           transforms.CenterCrop(int(args.img_size * 0.5)),
                                           transforms.Resize(args.img_size),
                                           transforms.ToTensor(),
                                           normalize])
    elif args.augmentation == "improved_v3":
        transform = DuplicatedCompose([transforms.RandomRotation(360, resample=Image.BILINEAR),
                                       transforms.CenterCrop(args.img_size),
                                       transforms.ToTensor(),
                                       normalize],
                                      [transforms.RandomAffine(20, translate=(0.1, 0.1), shear=10),
                                       transforms.RandomRotation(360, resample=Image.BILINEAR),
                                       transforms.CenterCrop(args.img_size),
                                       transforms.Grayscale(num_output_channels=3),
                                       transforms.ColorJitter(0.4, 0.4, 0.4, 0.125),
                                       transforms.ToTensor(), normalize])

        transform_val = DuplicatedCompose([transforms.CenterCrop(args.img_size),
                                           transforms.ToTensor(),
                                           normalize],
                                          [transforms.RandomAffine(20, translate=(0.1, 0.1), shear=10),
                                           transforms.RandomRotation(360, resample=Image.BILINEAR),
                                           transforms.CenterCrop(args.img_size),
                                           transforms.ToTensor(), normalize])

    elif args.augmentation == "improved_v3.1":
        transform = DuplicatedCompose([transforms.RandomRotation(360, resample=Image.BILINEAR),
                                       transforms.CenterCrop(args.img_size),
                                       transforms.ToTensor(),
                                       normalize],
                                      [transforms.RandomAffine(20, translate=(0.1, 0.1), shear=10),
                                       transforms.RandomRotation(360, resample=Image.BILINEAR),
                                       transforms.CenterCrop(args.img_size),
                                       transforms.RandomApply(torch.nn.ModuleList([transforms.Grayscale(num_output_channels=3)]), p=0.5),
                                       transforms.ColorJitter(0.4, 0.4, 0.4, 0.125),
                                       transforms.ToTensor(), normalize])

        transform_val = DuplicatedCompose([transforms.CenterCrop(args.img_size),
                                           transforms.ToTensor(),
                                           normalize],
                                          [transforms.RandomAffine(20, translate=(0.1, 0.1), shear=10),
                                           transforms.RandomRotation(360, resample=Image.BILINEAR),
                                           transforms.CenterCrop(args.img_size),
                                           transforms.ToTensor(), normalize])

    elif args.augmentation == "improved_v4":
        transform = DuplicatedCompose([transforms.RandomRotation(360, resample=Image.BILINEAR),
                                       transforms.CenterCrop(args.img_size),
                                       transforms.ToTensor(),
                                       normalize],
                                      [transforms.RandomAffine(20, translate=(0.1, 0.1), shear=10),
                                       transforms.RandomRotation(360, resample=Image.BILINEAR),
                                       transforms.CenterCrop(args.img_size),
                                       transforms.Grayscale(num_output_channels=3),
                                       transforms.GaussianBlur(1, sigma=(0.1, 2.0)),
                                       transforms.ColorJitter(0.4, 0.4, 0.4, 0.125),
                                       transforms.ToTensor(), normalize])

        transform_val = DuplicatedCompose([transforms.CenterCrop(args.img_size),
                                           transforms.ToTensor(),
                                           normalize],
                                          [transforms.RandomAffine(20, translate=(0.1, 0.1), shear=10),
                                           transforms.RandomRotation(360, resample=Image.BILINEAR),
                                           transforms.CenterCrop(args.img_size),
                                           transforms.ToTensor(), normalize])

    elif args.augmentation == "improved_v4.1":
        transform = DuplicatedCompose([transforms.RandomRotation(360, resample=Image.BILINEAR),
                                       transforms.CenterCrop(args.img_size),
                                       transforms.ToTensor(),
                                       normalize],
                                      [transforms.RandomAffine(20, translate=(0.1, 0.1), shear=10),
                                       transforms.RandomRotation(360, resample=Image.BILINEAR),
                                       transforms.CenterCrop(args.img_size),
                                       transforms.RandomApply(torch.nn.ModuleList([transforms.Grayscale(num_output_channels=3),transforms.GaussianBlur(1, sigma=(0.1, 2.0))]), p=0.5),
                                       transforms.ColorJitter(0.4, 0.4, 0.4, 0.125),
                                       transforms.ToTensor(), normalize])

        transform_val = DuplicatedCompose([transforms.CenterCrop(args.img_size),
                                           transforms.ToTensor(),
                                           normalize],
                                          [transforms.RandomAffine(20, translate=(0.1, 0.1), shear=10),
                                           transforms.RandomRotation(360, resample=Image.BILINEAR),
                                           transforms.CenterCrop(args.img_size),
                                           transforms.ToTensor(), normalize])

    elif args.augmentation == "improved_v5":
        transform = DuplicatedCompose([transforms.RandomRotation(360, resample=Image.BILINEAR),
                                       transforms.CenterCrop(args.img_size),
                                       transforms.ToTensor(),
                                       normalize],
                                      [transforms.RandomAffine(20, translate=(0.1, 0.1), shear=10),
                                       transforms.RandomRotation(360, resample=Image.BILINEAR),
                                       transforms.CenterCrop(args.img_size),
                                       transforms.Grayscale(num_output_channels=3),
                                       transforms.GaussianBlur(1, sigma=(0.1, 2.0)),
                                       transforms.ColorJitter(0.4, 0.4, 0.4, 0.125),
                                       transforms.ToTensor(), normalize,
                                       transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))])

        transform_val = DuplicatedCompose([transforms.CenterCrop(args.img_size),
                                           transforms.ToTensor(),
                                           normalize],
                                          [transforms.RandomAffine(20, translate=(0.1, 0.1), shear=10),
                                           transforms.RandomRotation(360, resample=Image.BILINEAR),
                                           transforms.CenterCrop(args.img_size),
                                           transforms.ToTensor(), normalize])
    img_train_dir = os.path.join(args.data_dir, args.data_type, 'train')
    img_test_dir = os.path.join(args.data_dir, args.data_type, 'test')

    args.img_train_dir = os.path.join(img_train_dir,args.data_folder)
    args.img_test_dir = os.path.join(img_test_dir,args.data_folder)

    train_dataset = ImageFolerRemap(img_train_dir,args.data_folder, transform=transform,args=args)
    train_with_idx = ImageFolerRemap(img_train_dir,args.data_folder,  transform=transform, with_idx=True,args=args)
    val_dataset = ImageFolerRemap(img_test_dir,args.data_folder,  transform=transform_val,args=args)

    classes, class_to_idx = train_dataset.find_classes(img_train_dir)
    data_idx = [class_to_idx[args.data_folder]]
    args.att_to_use = data_idx
    class_to_use = args.att_to_use
    remap_table = {}
    i = 0
    for k in data_idx:
        remap_table[k] = i
        i += 1


    tot_targets = torch.tensor(train_dataset.targets)
    val_targets = torch.tensor(val_dataset.targets)

    train_idx = None
    val_idx = None

    min_data = 99999999
    max_data = 0

    for k in class_to_use:
        tmp_idx = (tot_targets == k).nonzero()
        tmp_val_idx = (val_targets == k).nonzero()

        tot_train_tmp = len(tmp_idx)

        if min_data > tot_train_tmp:
            min_data = tot_train_tmp
        if max_data < tot_train_tmp:
            max_data = tot_train_tmp

        if k == class_to_use[0]:
            train_idx = tmp_idx.clone()
            val_idx = tmp_val_idx.clone()
        else:
            train_idx = torch.cat((train_idx, tmp_idx))
            val_idx = torch.cat((val_idx, tmp_val_idx))

    train_dataset_ = torch.utils.data.Subset(train_dataset, train_idx)
    val_dataset_ = torch.utils.data.Subset(val_dataset, val_idx)

    args.min_data = min_data
    args.max_data = max_data

    train_dataset = {'TRAIN': train_dataset_, 'FULL': train_dataset, 'IDX': train_with_idx}
    val_dataset = {'VAL': val_dataset_, 'FULL': val_dataset}






    return train_dataset, val_dataset



if __name__ == '__main__':
    print("dataset getter")
