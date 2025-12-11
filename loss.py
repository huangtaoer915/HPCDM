import torch.nn as nn
import torch
from torchvision import models

class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, x, y):
        diff = torch.fft.fft2(x.to('cuda:0')) - torch.fft.fft2(y.to('cuda:0'))
        loss = torch.mean(abs(diff))
        return loss
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
class AASGLoss(nn.Module):
    """
    Adaptive Amplitude Spectrum Guided Loss (AASG-Loss)

    对 haze 和 prompt 的幅度谱做 VGG19 感知约束：
    L_AASG = sum_i eta_i * || Phi_i(A(haze)) - Phi_i(A(prompt)) ||_1
    """

    def __init__(self, device="cuda", requires_grad=False):
        super(AASGLoss, self).__init__()
        self.vgg = Vgg19(requires_grad=requires_grad).to(device)
        self.l1 = nn.L1Loss()
        # 对应公式里的 η_i，和你 ContrastLoss 一致
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, haze, prompt):
        """
        haze:  B x C x H x W  (haze image / feature, 作为“物理参考”)
        prompt: B x C x H x W (生成的 prompt，对其回传梯度)
        """

        # 1) 计算幅度谱 A(haze), A(prompt)
        haze_f = torch.fft.fft2(haze)
        prompt_f = torch.fft.fft2(prompt)

        haze_amp = torch.abs(haze_f)
        prompt_amp = torch.abs(prompt_f)

        # 2) 输入 VGG 提取多层特征
        haze_feats = self.vgg(haze_amp)
        prompt_feats = self.vgg(prompt_amp)

        # 3) 分层加权 L1 感知损失（只对 prompt 回传梯度）
        loss = 0.0
        for i in range(len(haze_feats)):
            # haze 只作为参考，不回梯度
            loss_i = self.l1(haze_feats[i].detach(), prompt_feats[i])
            loss += self.weights[i] * loss_i

        return loss