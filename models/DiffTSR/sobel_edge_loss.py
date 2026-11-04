import torch
import torch.nn.functional as F
import torch.nn as nn

import math


class GaussianSmoothing(nn.Module):
    """
    Arguments:
    Apply gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed seperately for each channel in the input
    using a depthwise convolution.
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel. sigma (float, sequence): Standard deviation of the
        gaussian kernel. dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    # channels=1, kernel_size=kernel_size, sigma=sigma, dim=2
    def __init__(
        self,
        channels: int = 1,
        kernel_size: int = 3,
        sigma: float = 0.5,
        dim: int = 2,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, float):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError("Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim))

    def forward(self, input):
        """
        Arguments:
        Apply gaussian filter to input.
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups)

class DifferentiableEdgeDetector(nn.Module):
    def __init__(self, sigma=3.0, sobel_ksize=3):
        super().__init__()

        self.smth_3 = GaussianSmoothing(sigma=sigma).eval()
        self.smth_3.requires_grad_(False)

        # Sobel算子
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self._init_sobel(sobel_ksize)


    def _init_sobel(self, ksize):
        # X方向Sobel核
        sobel_x = torch.tensor([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]], dtype=torch.float32)

        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32)
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)

        self.sobel_x.weight = nn.Parameter(sobel_x)
        self.sobel_y.weight = nn.Parameter(sobel_y)
        self.sobel_x.weight.requires_grad_(False)
        self.sobel_y.weight.requires_grad_(False)

    def forward(self,x):
        weights = torch.tensor([0.299, 0.587, 0.114], device=x.device).view(1, 3, 1, 1)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        # print("输入张量形状:", x.shape)
        x = F.conv2d(x, weights)
        # print("转换为灰度后的张量形状:", x.shape)
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')
        x_blur = self.smth_3(x)  # 高斯模糊降噪
        # print("高斯模糊后的张量形状:", x_blur.shape)
        grad_x = self.sobel_x(x_blur).squeeze(1)[:,1:-1, 1:-1]
        grad_y = self.sobel_y(x_blur).squeeze(1)[:,1:-1, 1:-1]
        # print(grad_x.shape, grad_y.shape)
        # 添加小的epsilon值以避免数值不稳定
        epsilon = 1e-8
        edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + epsilon)
        # if edge_magnitude.numel() > 0:
        #     print(edge_magnitude.max(), edge_magnitude.min())
        # else:
        #     print("警告：edge_magnitude 张量为空！")

        # 添加梯度裁剪以防止梯度爆炸
        edge_magnitude = torch.clamp(edge_magnitude, 0, 255)

        return edge_magnitude  # [H,W]
