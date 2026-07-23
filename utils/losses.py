from csv import Error
import torch
import torch.nn.functional as F
import torch.nn as nn

import cv2
import numpy as np
import math

from models.recognizer import crop_image,adjust_image

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
        x = F.conv2d(x, weights)
        x = F.pad(x, (1, 1, 1, 1), mode='reflect')
        x_blur = self.smth_3(x)  # 高斯模糊降噪
        
        grad_x = self.sobel_x(x_blur).squeeze()[1:-1, 1:-1]
        grad_y = self.sobel_y(x_blur).squeeze()[1:-1, 1:-1]
        # 添加小的epsilon值以避免数值不稳定
        epsilon = 1e-8
        edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + epsilon) 

        # 添加梯度裁剪以防止梯度爆炸
        edge_magnitude = torch.clamp(edge_magnitude, 0, 255)
        
        return edge_magnitude  # [H,W]


def get_edge_loss(edge_detector, X, GT, Bbox_data,names, loss_type="l2"):
    all_edge_loss = []
    bsz = X.shape[0]      
    for batch in range(bsz):
        name = names[batch]
        try:
            bbs = Bbox_data[name]
        except:
            bbs = Bbox_data[name+".png"]
        if len(bbs) == 0:
            all_edge_loss += [torch.tensor(0.0).to(X.device)]
            continue
        # Create mask for each bounding box and crop the text regions
        bs_edge_loss =[]
        for bb in bbs:
            # Crop the text region
            x_text = adjust_image(bb,X[batch])
            gt_text = adjust_image(bb,GT[batch])
            x_edge = edge_detector(x_text)
            gt_edge = edge_detector(gt_text)
            # 检查边缘检测结果是否有效
            if torch.isnan(x_edge).any() or torch.isnan(gt_edge).any():
                raise Error("NaN detected in edge detection result")
            if loss_type == 'l1':
                loss_edge = (x_edge - gt_edge).abs()
            elif loss_type == 'l2':
                loss_edge = torch.nn.functional.mse_loss(gt_edge, x_edge, reduction="mean")
            else:
                raise NotImplementedError("unknown loss type '{loss_type}'")
            if torch.isnan(loss_edge) or loss_edge.item() > 1e6:  # 添加上限检查
                print(loss_edge)
                raise Error("NaN detected in loss computation")
            bs_edge_loss += [loss_edge]
        all_edge_loss += [torch.stack(bs_edge_loss).mean()]
    return torch.stack(all_edge_loss).mean()

def calc_ocr_loss(text_recognizer, X, GT, Bbox_data,names,loss_type="l2"):
    all_ocr_loss = torch.tensor(0.0).to(X.device)
    bsz = X.shape[0]
        
    cropped_texts = []
    cropped_gt = []
    for batch in range(bsz):
        x = X[batch]
        gt = GT[batch]
        name = names[batch]
        bbs = Bbox_data[name+".png"]
        if len(bbs) == 0:
            continue
        # Create mask for each bounding box and crop the text regions
        for bb in bbs:
            # Convert bounding box to appropriate format if needed
            bb = np.array(bb, dtype=np.int32)

            # Create mask with the polygon
            pos = np.zeros((gt.shape[1],gt.shape[2],1), dtype=np.uint8) # h,w,c
            # print(pos.shape)
            cv2.fillPoly(pos, [bb], (255, 255, 255))

            # Crop the text region
            x_text = crop_image(x, pos)
            gt_text = crop_image(gt, pos)
            cropped_texts.append(x_text)
            cropped_gt.append(gt_text)
        x_list = cropped_texts + cropped_gt

        # Get predictions from the recognizer
        preds, preds_neck = text_recognizer.pred_imglist(x_list, show_debug=False)

        n_pairs = len(preds)//2
        # Calculate OCR loss
        if n_pairs > 0:
            # OCR loss calculation

            preds_neck_x0 = preds_neck[:n_pairs]
            preds_neck_gt = preds_neck[n_pairs:]
            if loss_type == 'l1':
                loss_ocr = (preds_neck_gt - preds_neck_x0).abs()
            elif loss_type == 'l2':
                loss_ocr = torch.nn.functional.mse_loss(preds_neck_gt, preds_neck_x0, reduction='none')
            else:
                raise NotImplementedError("unknown loss type '{loss_type}'")
            all_ocr_loss += loss_ocr.mean()
        
    
    
    return all_ocr_loss
        

class L1WTVLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, target, seg, alpha):
        l1_loss = torch.nn.functional.l1_loss(pred, target, reduction='mean')

        grad_sr_x = F.pad(torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:]), pad=(0, 1, 0, 0))
        grad_sr_y = F.pad(torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :]), pad=(0, 0, 0, 1))

        grad_seg_x = F.pad(torch.abs(seg[:, :, :, :-1] - seg[:, :, :, 1:]), pad=(0, 1, 0, 0))
        grad_seg_y = F.pad(torch.abs(seg[:, :, :-1, :] - seg[:, :, 1:, :]), pad=(0, 0, 0, 1))

        grad_sr = torch.stack(tensors=[grad_sr_x, grad_sr_y], dim=0)
        grad_seg = torch.stack(tensors=[grad_seg_x, grad_seg_y], dim=0)

        wtv_loss = torch.mean(grad_seg / (grad_sr + 1e-5))

        ctx.save_for_backward(pred, target, grad_sr, grad_seg)
        ctx.alpha = alpha

        return l1_loss + alpha * wtv_loss

    @staticmethod
    def backward(ctx, grad_output):
        pred, target, grad_sr, grad_seg = ctx.saved_tensors
        alpha = ctx.alpha

        grad_l1 = torch.sign(pred - target) / pred.numel()

        grad_wtv = -grad_seg / ((grad_sr + 1e-5) ** 2)
        grad_wtv /= pred.numel()

        l1_max = grad_l1.abs().max()
        grad_wtv = torch.clamp(grad_wtv, min=-l1_max.item(), max=l1_max.item())

        grad_pred = grad_output * (grad_l1 + alpha * grad_wtv)

        return grad_pred, None, None, None


class L1WTVLoss(nn.Module):
    def __init__(self, loss_weight=1.0, wtv_weight=0.1):
        super().__init__()
        self.loss_weight = loss_weight
        self.wtv_weight = wtv_weight

    def forward(self, pred, target, seg):
        loss = L1WTVLossFunction.apply(pred, target, seg, self.wtv_weight)
        return self.loss_weight * loss
