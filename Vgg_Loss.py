from torchvision import models
import torch
import torch.nn as nn
import cv2
import os
import numpy as np
import torch.nn.functional as F


class VGG_Loss(nn.Module):
    # 不要忘记继承Module
    def __init__(self, device, band_width=0.5):
        super(VGG_Loss, self).__init__()
        self.vgg_model = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1).features[0:52].to(device)
        self.band_width = band_width
        self.register_buffer(
            name='vgg_mean',
            tensor=torch.tensor(
                [[[0.485]], [[0.456]], [[0.406]]], requires_grad=False,device=device)
        )
        self.register_buffer(
            name='vgg_std',
            tensor=torch.tensor(
                [[[0.229]], [[0.224]], [[0.225]]], requires_grad=False,device=device)
        )

    def compute_cx(self, dist_tilde, band_width):
        w = torch.exp((1 - dist_tilde) / band_width)  # Eq(3)
        cx = w / torch.sum(w, dim=2, keepdim=True)  # Eq(4)
        return cx

    def compute_relative_distance(self, dist_raw):
        dist_min, _ = torch.min(dist_raw, dim=2, keepdim=True)
        dist_tilde = dist_raw / (dist_min + 1e-5)
        return dist_tilde

    def compute_cosine_distance(self, x, y):
        # mean shifting by channel-wise mean of `y`.
        y_mu = y.mean(dim=(0, 2, 3), keepdim=True)
        x_centered = x - y_mu
        y_centered = y - y_mu

        # L2 normalization
        x_normalized = F.normalize(x_centered, p=2, dim=1)
        y_normalized = F.normalize(y_centered, p=2, dim=1)

        # channel-wise vectorization
        N, C, *_ = x.size()
        x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
        y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)

        # consine similarity

        cosine_sim = torch.bmm(x_normalized.transpose(1, 2),
                               y_normalized)  # (N, H*W, H*W)

        # convert to distance
        dist = 1 - cosine_sim

        return dist

    def forward(self, g, gt):
        """output和target都是1-D张量,换句话说,每个样例的返回是一个标量.
        """
        b, c, num_pic, h, w = g.shape

        g = g.permute(0, 2, 1, 3, 4).reshape(b * num_pic, c, h, w)
        gt = gt.permute(0, 2, 1, 3, 4).reshape(b * num_pic, c, h, w)

        # normalization
        g = g.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())
        gt = gt.sub(self.vgg_mean.detach()).div(self.vgg_std.detach())

        # picking up vgg feature maps
        g = self.vgg_model(g)
        gt = self.vgg_model(gt)

        assert g.size() == gt.size(), 'Vgg loss input tensor must have the same size.'
        dist_raw = self.compute_cosine_distance(g, gt)

        dist_tilde = self.compute_relative_distance(dist_raw)
        cx = self.compute_cx(dist_tilde, self.band_width)
        cx = torch.mean(torch.max(cx, dim=1)[0], dim=1)  # Eq(1)
        cx_loss = torch.mean(-torch.log(cx + 1e-5))  # Eq(5)

        return cx_loss


if __name__ == "__main__":
    model_vgg = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)
    model_ch = model_vgg.features[0:52]
    input = torch.randn(2, 3, 5, 144, 288)
    root = r"data_process_multiprocess\00001\00001"
    path_list = os.listdir(root)
    img = []
    for i in range(0, 50, 10):
        img.append(cv2.resize(cv2.imread(os.path.join(root, path_list[i])), (288, 288)))

    img_batch_1 = np.asarray(img) / 255.
    img_batch_1 = np.transpose(img_batch_1, (3, 0, 1, 2))
    img_batch_1 = torch.FloatTensor(img_batch_1).unsqueeze(0)

    root = r"data_process_multiprocess\00003\00007"
    path_list = os.listdir(root)
    img = []
    for i in range(0, 50, 10):
        img.append(cv2.resize(cv2.imread(os.path.join(root, path_list[i])), (288, 288)))
    img_batch_2 = np.asarray(img) / 255.
    img_batch_2 = np.transpose(img_batch_2, (3, 0, 1, 2))
    img_batch_2 = torch.FloatTensor(img_batch_2).unsqueeze(0)

    img_batch = torch.cat([img_batch_1, img_batch_2])

    img_v = img_batch.permute(0, 2, 1, 3, 4)
    img_total = img_v.view(3, 288, 288, 10)
    import matplotlib.pyplot as plt

    plt.imshow(img_total[:, :, :, 0].permute(1, 2, 0))
    plt.show()

    out = model_ch(input)

    import matplotlib.pyplot as plt

    plt.imshow(img_batch[:, :, 0:3])
    plt.show()
    print(out.shape)
    print(model_vgg)
    print(model_ch)
