import json
from torch.utils import data as data
from torchvision import transforms
from PIL import Image, ImageFilter
import cv2
import math
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as tF
import os
import torch.nn.functional as F
from data.utils import find_files_directory, img2tensor

def resize_and_cat(img, target_size):
    """
    等比例 resize 图像，并根据需要进行填充

    Args:
        img: 输入PIL Image
        target_size: 目标尺寸 (target_height, target_width)

    Returns:
        resized_img: 处理后的PIL Image
        scale_ratio: 缩放比例
        padding: 填充信息
        repeats: 重复次数
    """
    # 将PIL Image转换为张量
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img)
    
    C, H, W = img_tensor.shape
    target_h, target_w = target_size

    # 确定最短边并计算缩放比例
    short_side = H

    scale_ratio = 128.0 / short_side

    # 等比例缩放
    new_h = int(H * scale_ratio)
    new_w = int(W * scale_ratio)

    # 使用 interpolate 进行缩放
    resized_img = torch.nn.functional.interpolate(
        img_tensor.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
    ).squeeze(0)

    # 检查长宽比
    aspect_ratio = max(new_h, new_w) / min(new_h, new_w)

    if aspect_ratio > 4.0:
        # 如果长宽比 > 4:1，直接 resize 到目标尺寸
        final_img_tensor = torch.nn.functional.interpolate(
            resized_img.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
        ).squeeze(0)
        padding = (0, 0, 0, 0)
        repeats = 1
    else:
        # 如果长宽比 ≤ 4:1，通过循环 concat 填充
        repeats = target_w // new_w
        temp_img = resized_img
        for _ in range(repeats - 1):
            temp_img = torch.cat([temp_img, resized_img], dim=2)  # 在宽度方向拼接

        # 直接 resize 到目标尺寸，而不是截断
        final_img_tensor = torch.nn.functional.interpolate(
            temp_img.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False
        ).squeeze(0)

        # 计算填充信息
        # 因为我们是通过 resize 调整大小，填充信息需要相应调整
        # 原始拼接后的宽度
        concat_width = new_w * repeats
        # 计算填充的比例
        if concat_width !=0:
            padding_ratio = (concat_width - new_w) / concat_width
        else:
            padding_ratio = 0.0
        # 计算目标尺寸下的填充像素数
        padding_pixels = int(target_w * padding_ratio)
        padding = (0, 0, padding_pixels, 0)  # 左上右下

    # 将张量转换回PIL Image
    to_pil = transforms.ToPILImage()
    final_img = to_pil(final_img_tensor)
    
    return final_img, scale_ratio, padding, repeats

def resize_and_pad(img, new_shape=(640, 640), stride=32, auto=True, scaleFill=False, scaleup=True):
    """
    将输入的 PyTorch Tensor 图像 resize 并 pad 到指定尺寸
    
    参数:
        img: PyTorch Tensor, 形状为 [C, H, W] 或 [B, C, H, W]
        new_shape: 目标尺寸 (height, width)
        stride: 步长，用于确保填充后的尺寸是步长的倍数
        auto: 是否自动调整填充以满足步长要求
        scaleFill: 是否直接缩放到目标尺寸（不保持原始比例）
        scaleup: 是否允许放大图像
    
    返回:
        resized_img: 调整大小并填充后的图像，尺寸为 new_shape
        ratio: 缩放比例 (w_ratio, h_ratio)
        pad: 填充信息 (left, top, right, bottom)，用于后续去除填充
    """
    # 确保输入是 PyTorch Tensor
    if not isinstance(img, torch.Tensor):
        raise TypeError("Input must be a PyTorch Tensor")
    
    # 处理 batch 维度
    if len(img.shape) == 4:  # [B, C, H, W]
        B, C, H, W = img.shape
        batch_mode = True
    elif len(img.shape) == 3:  # [C, H, W]
        C, H, W = img.shape
        batch_mode = False
        img = img.unsqueeze(0)  # 添加 batch 维度 [1, C, H, W]
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    
    # 确保 new_shape 是元组
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # 计算目标高度和宽度
    target_h, target_w = new_shape
    
    # 如果需要，调整 target_h 和 target_w 使其为 stride 的倍数
    if auto and stride > 0:
        # 向上取整到 stride 的倍数
        target_h = ((target_h + stride - 1) // stride) * stride
        target_w = ((target_w + stride - 1) // stride) * stride
    
    if scaleFill:  # 拉伸填充，不保持宽高比
        # 直接调整到目标尺寸
        resized_img = F.interpolate(img, size=(target_h, target_w), mode='bilinear', align_corners=False)
        # 没有填充
        ratio = (target_w / W, target_h / H)  # 宽度、高度比例
        pad = (0, 0, 0, 0)  # 左上右下
        
        # 如果原始输入没有 batch 维度，则移除添加的 batch 维度
        if not batch_mode:
            resized_img = resized_img.squeeze(0)
        
        return resized_img, ratio, pad
    
    # 计算缩放比例，保持宽高比
    r = min(target_h / H, target_w / W)
    
    # 如果不允许放大且原图小于目标尺寸，则保持原始尺寸
    if not scaleup and r > 1:
        r = 1
    
    # 计算缩放后的尺寸
    new_h, new_w = int(round(H * r)), int(round(W * r))
    
    # 调整图像大小
    if (new_h, new_w) != (H, W):
        resized_img = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)
    else:
        resized_img = img.clone()
    
    # 计算需要填充的像素数
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    
    # 计算上下左右填充
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    # 执行填充
    resized_img = F.pad(resized_img, (left, right, top, bottom), value=0.0)
    
    # 如果原始输入没有 batch 维度，则移除添加的 batch 维度
    if not batch_mode:
        resized_img = resized_img.squeeze(0)
    
    # 返回调整大小并填充后的图像，以及用于恢复的信息
    ratio = (r, r)  # 宽度、高度比例相同
    pad = (left, top, right, bottom)  # 左上右下
    
    return resized_img, ratio, pad



class CTRDataset(data.Dataset):
    def __init__(self, dataset_folder: str,patch_width=512,patch_height=128):
        super().__init__()
        self.input_folder = dataset_folder
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.img_names = os.listdir(os.path.join(self.input_folder, "gt"))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]
        image_hq_path = os.path.join(self.input_folder, "gt", name)
        image_lq_path = os.path.join(self.input_folder, "lr", name)
        img_pil_hq = Image.open(image_hq_path).convert("RGB").resize(
            (self.patch_width, self.patch_height), Image.LANCZOS)
        img_pil_lq = Image.open(image_lq_path).convert("RGB").resize(
            (self.patch_width, self.patch_height), Image.LANCZOS)
        text = os.path.splitext(os.path.basename(image_hq_path))[0]

        img_hq = img2tensor([np.array(img_pil_hq) / 255.0],
                            bgr2rgb=False, float32=True)[0]
        img_lq = img2tensor([np.array(img_pil_lq) / 255.0],
                            bgr2rgb=False, float32=True)[0]
        example = dict()
        example['hq'] = img_hq
        example['lq'] = img_lq
        example['text'] = text
        return example


class CTRDatasetSynMask(CTRDataset):
    def __init__(self, dataset_folder: str,patch_width=512,patch_height=128):
        super().__init__(dataset_folder,patch_width,patch_height)
        json_path = os.path.join(dataset_folder,'filter.json')
        with open(json_path,"r") as f:
            self.annos = json.load(f)
        self.img_names = list(self.annos.keys())

    def __getitem__(self, idx):
        name = self.img_names[idx]+'.png'
        image_hq_path = os.path.join(self.input_folder, "gt", name)
        image_lq_path = os.path.join(self.input_folder, "lr", name)
        image_mask_path = os.path.join(self.input_folder, "mask", name)
        img_pil_mask = (
            Image.open(image_mask_path)
            .convert("RGB")
            .resize((self.patch_width, self.patch_height), Image.LANCZOS)
        )
        img_pil_hq = (
            Image.open(image_hq_path)
            .convert("RGB")
            .resize((self.patch_width, self.patch_height), Image.LANCZOS)
        )
        img_pil_lq = (
            Image.open(image_lq_path)
            .convert("RGB")
            .resize((self.patch_width, self.patch_height), Image.LANCZOS)
        )
        img_mask = img2tensor(
            [np.array(img_pil_mask) / 255.0], bgr2rgb=False, float32=True
        )[0]
        img_hq = img2tensor(
            [np.array(img_pil_hq) / 255.0], bgr2rgb=False, float32=True
        )[0]
        img_lq = img2tensor(
            [np.array(img_pil_lq) / 255.0], bgr2rgb=False, float32=True
        )[0]
        example = dict()
        example["hq"] = img_hq
        example["lq"] = img_lq
        example["mask"] = img_mask
        example["text"] = self.annos[self.img_names[idx]]
        return example

class CTRDatasetTextName(CTRDataset):

    def ck_hw(self, img):
        """Check if the image is in the correct height and width."""
        W,H = img.size
        repeats = 1
        if H/W>2:
            img = img.transpose(Image.ROTATE_270)
        if W/H<=4:
            img,_,_,repeats = resize_and_cat(img, (self.patch_width, self.patch_height))
        img = img.resize((self.patch_width, self.patch_height), Image.LANCZOS)
        return img,repeats

    def __getitem__(self, idx):
        name = self.img_names[idx]

        image_hq_path = os.path.join(self.input_folder, "gt", name)
        image_lq_path = os.path.join(self.input_folder, "lr", name)
        image_mask_path = os.path.join(self.input_folder, "mask", name)

        img_pil_mask = (
            Image.open(image_mask_path)
            .convert("RGB")
        )
        img_pil_mask,_ = self.ck_hw(img_pil_mask)
        img_pil_hq = (
            Image.open(image_hq_path)
            .convert("RGB")
        )
        img_pil_hq,_ = self.ck_hw(img_pil_hq)
        img_pil_lq = (
            Image.open(image_lq_path)
            .convert("RGB")
        )
        img_pil_lq,repeats = self.ck_hw(img_pil_lq)
        img_mask = img2tensor(
            [np.array(img_pil_mask) / 255.0], bgr2rgb=False, float32=True
        )[0]
        img_hq = img2tensor(
            [np.array(img_pil_hq) / 255.0], bgr2rgb=False, float32=True
        )[0]
        img_lq = img2tensor(
            [np.array(img_pil_lq) / 255.0], bgr2rgb=False, float32=True
        )[0]
        example = dict()
        example["hq"] = img_hq
        example["lq"] = img_lq
        example["mask"] = img_mask
        example["text"] = name[:-4]*repeats
        return example
    
class CTRDatasetTextNamePadBlack(CTRDatasetTextName):
    def ck_hw(self, img):
        """Check if the image is in the correct height and width."""
        W,H = img.size
        repeats = 1
        if H/W>2:
            img = img.transpose(Image.ROTATE_270)
        if W/H<=4:
            img,_,_ = resize_and_pad(img, (self.patch_width, self.patch_height))
        else:
            img = img.resize((self.patch_width, self.patch_height), Image.LANCZOS)
        # Pad to black
        img = tF.pad(img, (0, 0, self.patch_width - img.width, self.patch_height - img.height), fill=0)
        return img,repeats

class CTRDatasetSynMaskTest(CTRDatasetSynMask):

    def __init__(self, dataset_folder: str,patch_width=512,patch_height=128):
        super().__init__(dataset_folder,patch_width,patch_height)
        json_path = os.path.join(dataset_folder, "val.json")
        with open(json_path, "r") as f:
            self.annos = json.load(f)
        self.img_names = list(self.annos.keys())


class CTRDatasetTest(data.Dataset):
    def __init__(self, dataset_folder: str):
        super().__init__()
        self.input_folder = dataset_folder
        self.img_names = os.listdir(os.path.join(self.input_folder))

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        name = self.img_names[idx]
        lr_path = os.path.join(self.input_folder, name)
        lr_img = Image.open(lr_path).convert("RGB")
        lr_img = tF.to_tensor(lr_img)
        lr_img = tF.resize(lr_img, size=[128, 512])
        return {
            "text": name,
            "lr": lr_img,
        }
