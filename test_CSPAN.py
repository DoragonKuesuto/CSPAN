import argparse
import torch
import numpy as np

from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import TestSetLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from model_CSPAN import CSPANet
parser = argparse.ArgumentParser(description="Pytorch CSPAN Eval")
parser.add_argument("--model_sam", type=str, default=".pth", help="model path")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--scale", type=str, default=4, help="upscale factor")
parser.add_argument('--testset_dir', type=str, default='.../test')
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")


opt = parser.parse_args()
if opt.cuda:
    torch.cuda.set_device(0)
def valid_sam(testing_data_loader,  model):
    psnr_epoch = 0
    ssim_epoch = 0
    for iteration, (HR_left, _, LR_left, LR_right) in enumerate(testing_data_loader):
        LR_left, LR_right, HR_left = LR_left / 255, LR_right / 255, HR_left / 255
        input_l, input_r, target = Variable(LR_left), Variable(LR_right), Variable(HR_left)
        print(input_l.shape)
        if opt.cuda:
            input_l = input_l.cuda()
            input_r = input_r.cuda()
            target = target.cuda()
        SR_list, _, _, _ = model(input_l, input_r)
        HR = SR_list[2]
        SR_left_np = np.array(torch.squeeze(HR[:, :, :, :].data.cpu(), 0).permute(1, 2, 0))
        HR_left_np = np.array(torch.squeeze(target[:, :, :, :].data.cpu(), 0).permute(1, 2, 0))
        PSNR = psnr(HR_left_np, SR_left_np)
        SSIM = ssim(HR_left_np, SR_left_np, multichannel=True)
        psnr_epoch = psnr_epoch + PSNR
        ssim_epoch =ssim_epoch + SSIM
    print("===> CSPANet_SAM Avg. PSNR: {:.8f} dB SSIM: {:.8f} dB".format(psnr_epoch/(iteration+1), ssim_epoch/(iteration+1)))

def valid(testing_data_loader, model):
    psnr_epoch = 0
    ssim_epoch = 0
    for iteration, (HR_left, _, LR_left, _) in enumerate(testing_data_loader):
        LR_left, HR_left = LR_left / 255, HR_left / 255
        input_l,  target_l = Variable(LR_left), Variable(HR_left)
        if opt.cuda:
            input_l = input_l.cuda()
            target_l = target_l.cuda()
        HR_l = model(input_l)
        SR_left_np = np.array(torch.squeeze(HR_l[:, :, :, 64:].data.cpu(), 0).permute(1, 2, 0))
        HR_left_np = np.array(torch.squeeze(target_l[:, :, :, 64:].data.cpu(), 0).permute(1, 2, 0))
        PSNR = psnr(HR_left_np, SR_left_np)
        SSIM = ssim(HR_left_np, SR_left_np, multichannel=True)
        psnr_epoch = psnr_epoch + PSNR
        ssim_epoch = ssim_epoch + SSIM
    print("===> SRResNet Avg. PSNR: {:.8f} dB SSIM: {:.8f} dB".format(psnr_epoch/(iteration+1), ssim_epoch/(iteration+1)))

def main():
    weights_sam = torch.load(opt.model_sam)
    model_sam = CSPANet()
    model_sam.load_state_dict(weights_sam['model'])
    if opt.cuda:
        model_sam.cuda()
    test_set1 = TestSetLoader(dataset_dir=opt.testset_dir + '/' + "Middlebury", scale_factor=opt.scale)
    test_loader1 = DataLoader(dataset=test_set1, num_workers=1, batch_size=1, shuffle=False)
    test_set2 = TestSetLoader(dataset_dir=opt.testset_dir + '/' + "KITTI2012", scale_factor=opt.scale)
    test_loader2 = DataLoader(dataset=test_set2, num_workers=1, batch_size=1, shuffle=False)
    test_set3 = TestSetLoader(dataset_dir=opt.testset_dir + '/' + "KITTI2015", scale_factor=opt.scale)
    test_loader3 = DataLoader(dataset=test_set3, num_workers=1, batch_size=1, shuffle=False)

    import datetime
    oldtime = datetime.datetime.now()
    # valid(test_loader, model)
    wotime = datetime.datetime.now()
    print('Time consuming: ', wotime - oldtime)
    print("Middlebury")
    valid_sam(test_loader1, model_sam)
    print("KITTI2012")
    valid_sam(test_loader2, model_sam)
    print("KITTI2015")
    valid_sam(test_loader3, model_sam)

def cal_psnr(img1, img2):
    img1 = img1.cpu()
    img2 = img2.cpu()
    img1_np = img1.detach().numpy()
    img2_np = img2.detach().numpy()
    return psnr(img1_np, img2_np)

if __name__ == '__main__':
    main()
