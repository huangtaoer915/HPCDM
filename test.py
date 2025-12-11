from model import Model
from option import *
from data_utils import get_dataloader
from Net import Net as Net
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loader,test_loader = get_dataloader(opt)
model = Model(Net,opt)
model.load_network()
with torch.no_grad():
    psnr,ssim,lpips = model.test(test_loader)
    print("psnr:",psnr,"ssim:",ssim,"lpips:",lpips)
