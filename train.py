from model import Model
from option import *
from data_utils import get_dataloader
from tensorboardX import SummaryWriter
from Net import Net as Net
import torch

import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loader, test_loader = get_dataloader(opt)
model = Model(Net, opt)
writer = SummaryWriter(log_dir=opt.logdir)


log_file = os.path.join(opt.logdir, 'metrics.txt')
os.makedirs(opt.logdir, exist_ok=True)


best_epoch = -1
best_psnr = 0
best_ssim = 0

best_psnr_only = 0
best_psnr_epoch = -1
best_psnr_ssim = 0

best_ssim_only = 0
best_ssim_epoch = -1
best_ssim_psnr = 0
start_epoch = 0

if os.path.exists(opt.model_loadPath) and os.path.exists(opt.opt_loadPath):
    print("Loading checkpoint model...")
    model.load_state_dict(torch.load(opt.model_loadPath))
    checkpoint = torch.load(opt.opt_loadPath)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    best_psnr = checkpoint["best_psnr"]
    best_ssim = checkpoint["best_ssim"]
    best_lpips = checkpoint["best_lpips"]
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resuming training from epoch {start_epoch}")
else:
    print("Starting training from scratch")

for epoch in range(start_epoch, opt.total_epoch):
    model.scheduler.step()
    lr = model.scheduler.get_last_lr()[0]
    loss = model.optimize_parameters(train_loader, epoch)

    with torch.no_grad():
        psnr, ssim, lpips = model.test(test_loader)
        print(f"Epoch {epoch}  PSNR: {psnr:.4f}  SSIM: {ssim:.4f}  lpips: {lpips:.4f}")

        writer.add_scalar('lr', lr, epoch)
        writer.add_scalar('psnr', psnr, epoch)
        writer.add_scalar('ssim', ssim, epoch)
        writer.add_scalar('lpips', lpips, epoch)
        writer.add_scalar('train_loss', loss, epoch)

        model.save_network(epoch, psnr, ssim)
        

        with open(log_file, 'a') as f:
            f.write(f"epoch:{epoch}\t\tssim:{ssim:.4f}\t\tpsnr:{psnr:.4f}\t\tlpips:{lpips:.4f}\n")

        if psnr > best_psnr_only:
            best_psnr_only = psnr
            best_psnr_epoch = epoch
            best_psnr_ssim = ssim

        if ssim > best_ssim_only:
            best_ssim_only = ssim
            best_ssim_epoch = epoch
            best_ssim_psnr = psnr

        if psnr > best_psnr and ssim > best_ssim:
            best_psnr = psnr
            best_ssim = ssim
            best_epoch = epoch

        print(f"Current Best PSNR: {best_psnr:.4f} (Epoch {best_epoch}), "
              f"Current Best SSIM: {best_ssim:.4f} (Epoch {best_epoch})")
        print(f"Best PSNR Ever: {best_psnr_only:.4f} (Epoch {best_psnr_epoch}), SSIM: {best_psnr_ssim:.4f}")
        print(f"Best SSIM Ever: {best_ssim_only:.4f} (Epoch {best_ssim_epoch}), PSNR: {best_ssim_psnr:.4f}")
        print("-" * 80)

writer.close()

print("\nTraining Summary:")
print(f"The best combination of PSNR and SSIM occurred at Epoch {best_epoch}, PSNR = {best_psnr:.4f}, SSIM = {best_ssim:.4f}")
print(f"Highest PSNR achieved at Epoch {best_psnr_epoch}, PSNR = {best_psnr_only:.4f}, SSIM = {best_psnr_ssim:.4f}")
print(f"Highest SSIM achieved at Epoch {best_ssim_epoch}, SSIM = {best_ssim_only:.4f}, PSNR = {best_ssim_psnr:.4f}")