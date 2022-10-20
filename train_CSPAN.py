import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import *
from numpy import clip
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from model_CSPAN import CSPANet
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# Training settings
parser = argparse.ArgumentParser(description="PyTorch CSPAN")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=300, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=0.0001")
parser.add_argument('--gamma', type=float, default=0.5, help='')
parser.add_argument("--step", type=int, default=35, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to pretrained SISR model (default: none)")
parser.add_argument('--pretrained', default='', type=str, help='default: none')
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.1, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--scale", default=4, type=int, help="upscale factor (default: 4)")
parser.add_argument('--testset_dir', type=str, default='.../test')


torch.cuda.set_device(0)

def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)
    opt.seed =random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = TrainSetLoader('.../Flickr1024_patches', opt.scale)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    print("===> Building model")
    model = CSPANet(nfeats = 64, factor = 4)

    print("===> Setting GPU")
    if opt.cuda:
        model = model.cuda()

    # optionally train from a pretrained SISR
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading checkpoint '{}'".format(opt.pretrained))
            model_dict = model.state_dict()
            pretrained_dict = torch.load(opt.pretrained)['model'].state_dict()
            pretrained_dict1 = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict1)
            model.load_state_dict(model_dict)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading model '{}'".format(opt.resume))
            weights = torch.load(opt.resume)
            opt.start_epoch = weights["epoch"] + 1
            model.load_state_dict(weights['model'])
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer = optim.Adam([paras for paras in model.parameters() if paras.requires_grad == True], lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, epoch, scheduler)

def train(training_data_loader, optimizer, model, epoch, scheduler):
    scheduler.step()
    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()

    loss_epoch = 0.0
    psnr_epoch = 0.0
    criterion_L1 = nn.L1Loss()
    criterion_mse = nn.MSELoss(size_average=False)

    for iteration, (HR_left, HR_right, LR_left, LR_right) in enumerate(training_data_loader):
        LR_left, LR_right, HR_left, HR_right = LR_left / 255, LR_right / 255, HR_left / 255, HR_right / 255
        b, c, h, w = LR_left.shape
        LR_left, LR_right, HR_left, HR_right = Variable(LR_left), Variable(LR_right), Variable(HR_left), Variable(HR_right)
        if opt.cuda:
            LR_left = LR_left.cuda()
            LR_right = LR_right.cuda()
            HR_left = HR_left.cuda()
            criterion_mse = criterion_mse.cuda()
            criterion_L1 = criterion_L1.cuda()
        SR_list, mid_features, map, mask = model(LR_left, LR_right)
        (M_right_to_left0, M_left_to_right0) = map
        (V_right_to_left0, V_left_to_right0) = mask
        loss_SR = 0
        for i in range(len(SR_list)):
            loss_SR += criterion_L1(SR_list[i], HR_left)

        LR_right_warped = torch.bmm(M_right_to_left0.contiguous().view(b * h, w, w),
                                    LR_right.permute(0, 2, 3, 1).contiguous().view(b * h, w, c))
        LR_right_warped = LR_right_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
        LR_left_warped = torch.bmm(M_left_to_right0.contiguous().view(b * h, w, w),
                                   LR_left.permute(0, 2, 3, 1).contiguous().view(b * h, w, c))
        LR_left_warped = LR_left_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)

        loss_photo0 = criterion_L1(LR_left * V_left_to_right0, LR_right_warped * V_left_to_right0) + \
                      criterion_L1(LR_right * V_right_to_left0, LR_left_warped * V_right_to_left0)

        loss = loss_SR + 0.01 * loss_photo0


        loss_epoch = loss_epoch + loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),opt.clip)
        optimizer.step()
        PSNR = cal_psnr(HR_left, SR_list[1])
        psnr_epoch = psnr_epoch + PSNR
        psnr_Avg = psnr_epoch / (iteration + 1)
    print("===> Epoch[{}]: Loss: {:.4f} PSNR: {:.4f} ".format(epoch, loss_epoch/(iteration+1), psnr_Avg))
    save_checkpoint_SAM(model, epoch,psnr_Avg)



def save_checkpoint_SAM(model, epoch,psnr_Avg):
    model_out_path = "checkpoint/" + "modelcover_epoch_{}_{:.3f}.pth".format(epoch,psnr_Avg)
    state = {"epoch": epoch ,"model": model.state_dict()}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

    #test
    weights_sam = torch.load(model_out_path)
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

    wotime = datetime.datetime.now()
    print('Time consuming: ', wotime - oldtime)
    print("Middlebury")
    valid_sam(test_loader1, model_sam)
    print("KITTI2012")
    valid_sam(test_loader2, model_sam)
    print("KITTI2015")
    valid_sam(test_loader3, model_sam)

def valid_sam(testing_data_loader,  model):
    psnr_epoch = 0
    ssim_epoch = 0
    for iteration, (HR_left, _, LR_left, LR_right) in enumerate(testing_data_loader):
        LR_left, LR_right, HR_left = LR_left / 255, LR_right / 255, HR_left / 255
        input_l, input_r, target = Variable(LR_left), Variable(LR_right), Variable(HR_left)
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
    print("===>  Avg. PSNR: {:.4f} dB SSIM: {:.5f} dB".format(psnr_epoch/(iteration+1), ssim_epoch/(iteration+1)))

def show(img):
    img = clip(img.data.cpu(), 0, 1)
    img = ToPILImage()(img[0,:,:,:])
    plt.figure(), plt.imshow(img)

def cal_psnr(img1, img2):
    img1 = img1.cpu()
    img2 = img2.cpu()
    img1_np = img1.detach().numpy()
    img2_np = img2.detach().numpy()
    return psnr(img1_np, img2_np)

if __name__ == "__main__":
    main()
