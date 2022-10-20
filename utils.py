from PIL import Image
import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import torch
import numpy as np

#Train调用：数据集加载

#验证集集加载调用
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG','.bmp'])
#01.训练集加载
class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        self.upscale_factor = upscale_factor
        self.path = '/patches_x' + str(upscale_factor) + '/'
        self.file_list = os.listdir(dataset_dir + self.path)
        # self.upscale_factor = upscale_factor
        self.tranform = Compose([
            augumentation(),
        ])
    def __getitem__(self, index):
        hr_image_left = Image.open(self.dataset_dir + self.path  + self.file_list[index] + '/hr0.png')
        hr_image_right = Image.open(self.dataset_dir + self.path + self.file_list[index] + '/hr1.png')
        lr_image_left = Image.open(self.dataset_dir + self.path + self.file_list[index] +  '/lr0.png')
        lr_image_right = Image.open(self.dataset_dir + self.path + self.file_list[index] + '/lr1.png')
        hr_image_left = np.array(hr_image_left, dtype=np.float32)
        hr_image_right = np.array(hr_image_right, dtype=np.float32)
        lr_image_left = np.array(lr_image_left, dtype=np.float32)
        lr_image_right = np.array(lr_image_right, dtype=np.float32)

        hr_image_left, hr_image_right, lr_image_left, lr_image_right = self.tranform(hr_image_left, hr_image_right,
                                                                                     lr_image_left, lr_image_right)
        hr_image_left, hr_image_right, lr_image_left, lr_image_right = ndarray2tensor()(hr_image_left, hr_image_right,
                                                                                        lr_image_left, lr_image_right)
        return hr_image_left, hr_image_right, lr_image_left, lr_image_right
    def __len__(self):
        return len(self.file_list)
#01.验证集集加载 ，未用到
class ValSetLoader(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValSetLoader, self).__init__()
        self.upscale_factor = upscale_factor
        self.hr_filenames = [os.path.join(dataset_dir + '/hr/', x) for x in os.listdir(dataset_dir) if is_image_file(x)]
        self.lr_filenames = [os.path.join(dataset_dir + '/lr/', x) for x in os.listdir(dataset_dir) if is_image_file(x)]
    def __getitem__(self, index):
        hr_image = Image.open(self.hr_filenames[index])
        lr_image = Image.open(self.lr_filenames[index])
        return ToTensor()(hr_image), ToTensor()(lr_image)
#01.测试集加载 valid用到  vaild才是test，代码中的test没用到
class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, scale_factor):
        super(TestSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        self.scale_factor = scale_factor
        self.file_list = os.listdir(dataset_dir + '/hr')
    def __getitem__(self, index):
        hr_image_left  = Image.open(self.dataset_dir + '/hr/' + self.file_list[index] + '/hr0.png')
        hr_image_right = Image.open(self.dataset_dir + '/hr/' + self.file_list[index] + '/hr1.png')
        lr_image_left  = Image.open(self.dataset_dir + '/lr_x' + str(self.scale_factor) + '/' + self.file_list[index] + '/lr0.png')
        lr_image_right = Image.open(self.dataset_dir + '/lr_x' + str(self.scale_factor) + '/' + self.file_list[index] + '/lr1.png')
        hr_image_left = np.array(hr_image_left, dtype=np.float32)
        hr_image_right = np.array(hr_image_right, dtype=np.float32)
        lr_image_left = np.array(lr_image_left, dtype=np.float32)
        lr_image_right = np.array(lr_image_right, dtype=np.float32)

        hr_image_left, hr_image_right, lr_image_left, lr_image_right = ndarray2tensor()(hr_image_left, hr_image_right,
                                                                                        lr_image_left, lr_image_right)
        return hr_image_left, hr_image_right, lr_image_left, lr_image_right
    def __len__(self):
        return len(self.file_list)

#暂未用到
class random_crop(object):
    def __init__(self, crop_size, upscale_factor):
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
    def __call__(self, hr_image_left, hr_image_right, lr_image_left, lr_image_right):
        lr_image_left = np.array(lr_image_left, dtype=np.float32)
        lr_image_right = np.array(lr_image_right, dtype=np.float32)
        hr_image_left = np.array(hr_image_left, dtype=np.float32)
        hr_image_right = np.array(hr_image_right, dtype=np.float32)
        h, w, _ = lr_image_left.shape
        start_x_input = random.randint(1, h-self.crop_size[0]-1)
        start_y_input = random.randint(1, w-self.crop_size[1]-1)
        start_x_target = start_x_input * self.upscale_factor
        start_y_target = start_y_input * self.upscale_factor

        lr_image_left = lr_image_left[start_x_input: start_x_input + self.crop_size[0], start_y_input: start_y_input + self.crop_size[1], :]
        lr_image_right = lr_image_right[start_x_input: start_x_input + self.crop_size[0],
                        start_y_input: start_y_input + self.crop_size[1], :]
        hr_image_left = hr_image_left[start_x_target: start_x_target + self.crop_size[0] * self.upscale_factor,
                        start_y_target: start_y_target + self.crop_size[1] * self.upscale_factor, :]
        hr_image_right = hr_image_right[start_x_target: start_x_target + self.crop_size[0] * self.upscale_factor,
                        start_y_target: start_y_target + self.crop_size[1] * self.upscale_factor, :]
        return hr_image_left, hr_image_right, lr_image_left, lr_image_right
#上面TrainSetLoader.__init__(),数据增强：水平翻转，垂直翻转，（通道混合暂无，抄NAFSSR的代码）
class augumentation(object):
    def __call__(self, hr_image_left, hr_image_right, lr_image_left, lr_image_right):
        if random.random()<0.5: # flip horizonly  水平翻转    50%概率
            lr_image_left = lr_image_left[:, ::-1, :]
            lr_image_right = lr_image_right[:, ::-1, :]
            hr_image_left = hr_image_left[:, ::-1, :]
            hr_image_right = hr_image_right[:, ::-1, :]
        if random.random()<0.5: #flip vertically 垂直翻转     50%概率
            lr_image_left = lr_image_left[::-1, :, :]
            lr_image_right = lr_image_right[::-1, :, :]
            hr_image_left = hr_image_left[::-1, :, :]
            hr_image_right = hr_image_right[::-1, :, :]
        """"no rotation
        if random.random()<0.5:
            lr_image_left = lr_image_left.transpose(1, 0, 2)
            lr_image_right = lr_image_right.transpose(1, 0, 2)
            hr_image_left = hr_image_left.transpose(1, 0, 2)
            hr_image_right = hr_image_right.transpose(1, 0, 2)
        """
        return hr_image_left, hr_image_right, lr_image_left, lr_image_right
#上面的TrainSetLoader/TestSetLoader的__getitem__()
class ndarray2tensor(object):
    def __init__(self):
        self.totensor = ToTensor()
    def __call__(self, hr_image_left, hr_image_right, lr_image_left, lr_image_right):
        lr_image_left = self.totensor(lr_image_left.copy())
        lr_image_right = self.totensor(lr_image_right.copy())
        hr_image_left = self.totensor(hr_image_left.copy())
        hr_image_right = self.totensor(hr_image_right.copy())
        return hr_image_left, hr_image_right, lr_image_left, lr_image_right
#未被调用
class L1Loss(object):
    def __call__(self, input, target):
        return torch.abs(input - target).mean()

#train的valid()调用
def rgb2y(img):
    img_r = img[:, 0, :, :]
    img_g = img[:, 1, :, :]
    img_b = img[:, 2, :, :]
    image_y = torch.round(0.257 * torch.unsqueeze(img_r, 1) + 0.504 * torch.unsqueeze(img_g, 1) + 0.098 * torch.unsqueeze(img_b, 1) + 16)
    return image_y
#train的init()调用
class Compose(object):
    def __init__(self, co_transforms):
        self.co_transforms = co_transforms
    def __call__(self, hr_image_left, hr_image_right, lr_image_left, lr_image_right):
        for transform in self.co_transforms:
            hr_image_left, hr_image_right, lr_image_left, lr_image_right = transform(hr_image_left, hr_image_right, lr_image_left, lr_image_right)
        return hr_image_left, hr_image_right, lr_image_left, lr_image_right
