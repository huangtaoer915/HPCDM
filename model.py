from tqdm import tqdm
import os
from torch import optim
from pytorch_msssim import *
from metrics import *
import torch
import torch.nn as nn
import numpy as np
from loss import fftLoss,AASGLoss

class Model(nn.Module):
    def __init__(self, Net, opts):
        super().__init__()
        self.opt = opts
        self.start_epoch = 0
        self.best_psnr = 0
        self.best_ssim = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.scaler = torch.cuda.amp.GradScaler()
        self.model = Net().to(self.device)
        self.optimizer = optim.AdamW(params=filter(lambda x: x.requires_grad, self.model.parameters()), lr=opts.lr,
                                     betas=(0.9, 0.999),
                                     eps=1e-08, amsgrad=False, weight_decay=0.01)

        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1 if epoch < self.opt.lr_list[0] else (0.5 if epoch < self.opt.lr_list[1] else 0.25))

        self.set_seed(opts.seed)
        self.fft_loss=fftLoss()
        self.L1Loss=nn.SmoothL1Loss()
        self.AASG_loss=AASGLoss()

    def set_seed(self, seed):
        seed = int(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    def save_network(self, epoch, now_psnr, now_ssim):
        if self.best_psnr < now_psnr and self.best_ssim < now_ssim:
            self.best_psnr = now_psnr
            self.best_ssim = now_ssim
            print("best_psnr:",self.best_psnr)
            print("best_ssim:",self.best_ssim)
            model_path = os.path.join(self.opt.model_Savepath, 'best_model.pth')
            opt_path = os.path.join(self.opt.optim_Savepath, 'best_opt.pth')
        elif epoch % self.opt.save_fre_step == 0:
            model_path = os.path.join(self.opt.model_Savepath, 'E{}_model.pth'.format(epoch))
            opt_path = os.path.join(self.opt.optim_Savepath, 'E{}_opt.pth'.format(epoch))
        else:
            return

        # model_save
        torch.save(
            self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
            model_path)
        # optim_save
        opt_state = {'epoch': epoch,
                     'best_ssim': self.best_ssim,
                     'best_psnr': self.best_psnr,
                     'scheduler': self.scheduler.state_dict(),
                     'optimizer': self.optimizer.state_dict()}
        torch.save(opt_state, opt_path)

    def load_network(self):
        model_path = self.opt.model_loadPath
        opt_path = self.opt.opt_loadPath
        self.model.load_state_dict(torch.load(model_path))
        optim_state = torch.load(opt_path)
        self.start_epoch = optim_state['epoch']
        self.best_psnr = optim_state['best_psnr']
        self.best_ssim = optim_state['best_ssim']
        self.optimizer.load_state_dict(optim_state['optimizer'])
        self.scheduler.load_state_dict(optim_state['scheduler'])
    # 需要根据net修改的部分
    def optimize_parameters(self, train_dataloader,epoch):
        self.model.train()
        total_loss = 0.0
        for idx, (input_img, label_img) in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch}', leave=True)):
            input_img = input_img.to(self.device)
            label_img = label_img.to(self.device)

            # 11111111111111111111111111111111111111111111111111111111111111111
            pred_img,prompt = self.model(input_img)
            fft_loss=self.fft_loss(pred_img, label_img)
            smoothL1_loss = self.L1Loss(pred_img, label_img)
            AASG_loss=self.amplitude_loss(input_img,prompt)

            # 负 log likelihood 为损失
            loss = smoothL1_loss + (fft_loss +AASG_loss)*0.01
            loss.backward()
            total_loss += loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        return total_loss / len(train_dataloader)

    def test(self, test_dataloder):
        self.model.eval()
        # torch.cuda.empty_cache()
        ssims = []
        psnrs = []
        lpips = []
        for step, (inputs, targets) in enumerate(tqdm(test_dataloder)):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            pred,prompt= self.model(inputs)
            ssims.append(ssim(pred, targets).item())
            psnrs.append(psnr(pred, targets))
            lpips.append(Lpips(pred, targets))
        ssim_mean = np.mean(ssims)
        psnr_mean = np.mean(psnrs)
        lpips_mean = np.mean(lpips)
        return psnr_mean, ssim_mean,lpips_mean

    def print_model(self):
        from thop import profile
        dummy_input = torch.randn(1, 3, 512, 512).to(self.device)
        flops, params = profile(self.model, (dummy_input,))
        print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))


