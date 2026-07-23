import math
import torch
import torch.nn.functional as F

def resize_and_cat(img, target_size):
    """
    等比例 resize 图像，并根据需要进行填充

    Args:
        img: 输入图像张量，形状为 [C, H, W]
        target_size: 目标尺寸 (target_height, target_width)

    Returns:
        resized_img: 处理后的图像
        scale_ratio: 缩放比例
        padding: 填充信息
    """
    C, H, W = img.shape
    target_h, target_w = target_size

    # 确定最短边并计算缩放比例
    short_side = H

    scale_ratio = 128.0 / short_side

    # 等比例缩放
    new_h = int(H * scale_ratio)
    new_w = int(W * scale_ratio)

    # 使用 interpolate 进行缩放
    resized_img = torch.nn.functional.interpolate(
        img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
    ).squeeze(0)

    # 检查长宽比
    aspect_ratio = max(new_h, new_w) / min(new_h, new_w)

    if aspect_ratio > 4.0:
        # 如果长宽比 > 4:1，直接 resize 到目标尺寸
        final_img = torch.nn.functional.interpolate(
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
        final_img = torch.nn.functional.interpolate(
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

def remove_padding(img, pad):
    """
    从调整大小并填充后的图像中移除填充部分
    
    参数:
        img: PyTorch Tensor, 形状为 [C, H, W] 或 [B, C, H, W]
        pad: 填充信息 (left, top, right, bottom)
    
    返回:
        unpadded_img: 移除填充后的图像
    """
    # 确保输入是 PyTorch Tensor
    if not isinstance(img, torch.Tensor):
        raise TypeError("Input must be a PyTorch Tensor")

    # 处理 batch 维度
    if len(img.shape) == 4:  # [B, C, H, W]
        batch_mode = True
    elif len(img.shape) == 3:  # [C, H, W]
        batch_mode = False
        img = img.unsqueeze(0)  # 添加 batch 维度
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    # 解包填充信息
    left, top, right, bottom = pad

    # 计算裁剪区域
    _, _, H, W = img.shape

    # 执行裁剪以移除填充
    unpadded_img = img[:, :, top:H-bottom, left:W-right]

    # 如果原始输入没有 batch 维度，则移除添加的 batch 维度
    if not batch_mode:
        unpadded_img = unpadded_img.squeeze(0)

    return unpadded_img


def split_image_with_overlap(image, block_size=(512, 512), overlap=(64, 64)):
    """
    将图像分割成重叠的块

    参数:
    - image: BCHW格式的torch.Tensor
    - block_size: 每个块的大小(高度, 宽度)
    - overlap: 重叠区域的大小(高度方向重叠, 宽度方向重叠)

    返回:
    - blocks: 包含所有分块的列表
    - positions: 每个块在原图中的位置(左上角坐标)
    """
    # 获取图像尺寸
    B, C, H, W = image.shape

    # 解包块大小和重叠大小
    block_h, block_w = block_size
    overlap_h, overlap_w = overlap

    # 计算有效步长(stride)
    stride_h = block_h - overlap_h
    stride_w = block_w - overlap_w

    # 计算需要多少行多少列的块
    num_blocks_h = math.ceil((H - overlap_h) / stride_h)
    num_blocks_w = math.ceil((W - overlap_w) / stride_w)

    # 可能需要对图像进行填充，以确保最后一个块也是完整的
    pad_h = stride_h * num_blocks_h + overlap_h - H
    pad_w = stride_w * num_blocks_w + overlap_w - W

    if pad_h > 0 or pad_w > 0:
        # 使用反射填充，保持边缘像素的连续性
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")

    blocks = []
    positions = []

    # 遍历并提取每个块
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            # 计算当前块的左上角坐标
            top = i * stride_h
            left = j * stride_w

            # 提取当前块
            block = image[:, :, top : top + block_h, left : left + block_w]
            blocks.append(block)
            positions.append((top, left))

    return blocks, positions


def merge_blocks_with_overlap(
    blocks, positions, original_shape, block_size=(512, 512), overlap=(64, 64)
):
    """
    将重叠的块合并回原始图像

    参数:
    - blocks: 包含所有分块的列表
    - positions: 每个块在原图中的位置
    - original_shape: 原始图像的形状 (B, C, H, W)
    - block_size: 每个块的大小(高度, 宽度)
    - overlap: 重叠区域的大小(高度方向重叠, 宽度方向重叠)

    返回:
    - merged: 合并后的图像
    """
    B, C, H, W = original_shape

    # 解包块大小和重叠大小
    block_h, block_w = block_size
    overlap_h, overlap_w = overlap

    stride_h = block_h - overlap_h
    stride_w = block_w - overlap_w

    # 创建权重掩码，用于平滑重叠区域
    weight_mask = torch.ones((block_h, block_w), device=blocks[0].device)

    # 在重叠区域应用线性权重，使过渡更平滑
    for i in range(overlap_h):
        # 上边缘
        weight_mask[i, :] = i / overlap_h
        # 下边缘
        weight_mask[block_h - i - 1, :] = i / overlap_h

    for j in range(overlap_w):
        # 左边缘
        weight_mask[:, j] = weight_mask[:, j] * (j / overlap_w)
        # 右边缘
        weight_mask[:, block_w - j - 1] = weight_mask[:, block_w - j - 1] * (
            j / overlap_w
        )

    # 创建输出图像和权重累积图像
    result = torch.zeros(original_shape, device=blocks[0].device)
    weight_sum = torch.zeros((B, 1, H, W), device=blocks[0].device)

    # 将每个块放回原位置，使用权重掩码
    for block, (top, left) in zip(blocks, positions):
        # 确保不超出原图边界
        h_end = min(top + block_h, H)
        w_end = min(left + block_w, W)
        h_range = h_end - top
        w_range = w_end - left
        print(h_range,w_range)
        print("block shape", block.shape)
        print(block[:, :, :h_range, :w_range].shape)
        print(weight_mask[:h_range, :w_range].unsqueeze(0).unsqueeze(0).shape)
        # 应用权重掩码
        weighted_block = block[:, :, :h_range, :w_range] * weight_mask[
            :h_range, :w_range
        ].unsqueeze(0).unsqueeze(0)

        # 累加到结果图像
        result[:, :, top:h_end, left:w_end] += weighted_block
        weight_sum[:, :, top:h_end, left:w_end] += (
            weight_mask[:h_range, :w_range].unsqueeze(0).unsqueeze(0)
        )

    # 归一化，避免重叠区域的值过大
    # 添加小值避免除零错误
    epsilon = 1e-6
    result = result / (weight_sum + epsilon)

    return result[:, :, :H, :W]  # 裁剪回原始大小


def extract_overlapping_patches(img, patch_size, overlap):
    """
    img: Tensor of shape (B, C, H_up, W_up)
    patch_size: tuple (patch_H, patch_W)
    overlap: tuple (overlap_H, overlap_W)
    """
    patch_H, patch_W = patch_size
    overlap_H, overlap_W = overlap

    stride_H = patch_H - overlap_H
    stride_W = patch_W - overlap_W

    # Use unfold to extract sliding patches
    patches = img.unfold(2, patch_H, stride_H).unfold(
        3, patch_W, stride_W
    )  # Shape: (B, C, num_patches_H, num_patches_W, patch_H, patch_W)

    # Rearrange to (B * num_patches, C, patch_H, patch_W)
    B, C, nH, nW, pH, pW = patches.shape
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B * nH * nW, C, pH, pW)

    return patches


def combine_sr_patches_weighted(patches, out_shape, patch_size, overlap):
    B, C, H_out, W_out = out_shape
    H_patch, W_patch = patch_size
    stride_H = H_patch - overlap[0]
    stride_W = W_patch - overlap[1]

    output = torch.zeros((B, C, H_out, W_out), device=patches.device)
    weight_map = torch.zeros((B, 1, H_out, W_out), device=patches.device)

    # 构造融合权重（二维梯度 mask）
    h_weight = (
        torch.linspace(0, 1, steps=H_patch, device=patches.device)
        .unsqueeze(1)
        .expand(H_patch, W_patch)
    )
    w_weight = (
        torch.linspace(0, 1, steps=W_patch, device=patches.device)
        .unsqueeze(0)
        .expand(H_patch, W_patch)
    )
    weight = torch.minimum(h_weight, 1 - h_weight) * torch.minimum(
        w_weight, 1 - w_weight
    )  # 形成中间强、边缘弱的2D mask
    weight = weight.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H_patch, W_patch)

    idx = 0
    nH = (H_out - overlap[0]) // stride_H
    nW = (W_out - overlap[1]) // stride_W

    for i in range(nH):
        for j in range(nW):
            top = i * stride_H
            left = j * stride_W
            bottom = top + H_patch
            right = left + W_patch

            patch = patches[idx].unsqueeze(0)
            w = weight.expand_as(patch)

            output[:, :, top:bottom, left:right] += patch * w
            weight_map[:, :, top:bottom, left:right] += w[
                :1, :1, :, :
            ]  # 只加一次通道权重
            idx += 1

    output = output / torch.clamp(weight_map, min=1e-6)
    return output


def combine_mask_patches(patches, out_shape, patch_size, overlap):
    B, C, H_out, W_out = out_shape
    H_patch, W_patch = patch_size
    stride_H = H_patch - overlap[0]
    stride_W = W_patch - overlap[1]

    output = torch.zeros(
        (B, C, H_out, W_out), dtype=patches.dtype, device=patches.device
    )

    idx = 0
    nH = (H_out - overlap[0]) // stride_H
    nW = (W_out - overlap[1]) // stride_W

    for i in range(nH):
        for j in range(nW):
            top = i * stride_H
            left = j * stride_W
            bottom = top + H_patch
            right = left + W_patch

            output[:, :, top:bottom, left:right] = torch.maximum(
                output[:, :, top:bottom, left:right], patches[idx]
            )
            idx += 1

    return output

import torch
import random


def random_crop_tensors(tensor1, tensor2, crop_size):
    """
    对两个tensor进行相同位置的随机裁剪

    参数:
    - tensor1: 第一个tensor，形状为 [B, C, H, W]
    - tensor2: 第二个tensor，形状为 [B, C, H2, W2]
    - crop_size: 裁剪大小，可以是整数(正方形裁剪)或元组(h, w)

    返回:
    - cropped_tensor1: 裁剪后的第一个tensor
    - cropped_tensor2: 裁剪后的第二个tensor
    """
    # 确保两个tensor的batch size相同
    assert tensor1.shape[0] == tensor2.shape[0], "两个tensor的batch size必须相同"

    # 获取tensor的尺寸
    batch_size = tensor1.shape[0]
    _, _, h1, w1 = tensor1.shape
    _, _, h2, w2 = tensor2.shape

    # 处理crop_size参数
    if isinstance(crop_size, int):
        crop_h, crop_w = crop_size, crop_size
    else:
        crop_h, crop_w = crop_size

    # 确保裁剪尺寸不大于原始尺寸
    assert crop_h <= min(h1, h2), f"裁剪高度 {crop_h} 大于最小tensor高度 {min(h1, h2)}"
    assert crop_w <= min(w1, w2), f"裁剪宽度 {crop_w} 大于最小tensor宽度 {min(w1, w2)}"

    # 计算可能的起始位置范围
    max_h = min(h1, h2) - crop_h
    max_w = min(w1, w2) - crop_w

    # 为每个样本在batch中选择相同的随机裁剪位置
    crops = []
    for _ in range(batch_size):
        top = random.randint(0, max_h)
        left = random.randint(0, max_w)
        crops.append((top, left))

    # 应用裁剪
    cropped_tensor1 = torch.zeros(
        batch_size, tensor1.shape[1], crop_h, crop_w, device=tensor1.device
    )
    cropped_tensor2 = torch.zeros(
        batch_size, tensor2.shape[1], crop_h, crop_w, device=tensor2.device
    )

    for i, (top, left) in enumerate(crops):
        cropped_tensor1[i] = tensor1[i, :, top : top + crop_h, left : left + crop_w]
        cropped_tensor2[i] = tensor2[i, :, top : top + crop_h, left : left + crop_w]

    return cropped_tensor1, cropped_tensor2


def random_crop_multiple_images(images, crop_size):
    """
    对多个PIL图像进行相同位置的随机裁剪

    参数:
    - images: PIL图像列表
    - crop_size: 裁剪尺寸，可以是整数(正方形裁剪)或元组(height, width)

    返回:
    - cropped_images: 裁剪后的图像列表
    """
    # 获取所有图像的尺寸
    sizes = [img.size for img in images]
    widths = [w for w, h in sizes]
    heights = [h for w, h in sizes]

    # 处理crop_size参数
    if isinstance(crop_size, int):
        crop_height = crop_width = crop_size
    else:
        crop_height, crop_width = crop_size

    # 确保裁剪尺寸不大于任何一个原始尺寸
    assert crop_height <= min(heights), "裁剪高度大于图像高度"
    assert crop_width <= min(widths), "裁剪宽度大于图像宽度"

    # 计算可能的起始位置范围
    max_x = min(widths) - crop_width
    max_y = min(heights) - crop_height

    # 随机选择裁剪的起始位置
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    # 裁剪所有图像
    cropped_images = [
        img.crop((x, y, x + crop_width, y + crop_height)) for img in images
    ]

    return cropped_images

import torch


def split_long_image_tensor(image_tensor, target_ratio=4.0, overlap_ratio=0.2):
    """
    将长宽比大于6:1的图片tensor分割成多个4:1的块，带有重叠区域

    参数:
    image_tensor: 输入图片tensor，格式为BCHW (batch, channels, height, width)
    target_ratio: 目标长宽比，默认为4.0
    overlap_ratio: 重叠区域占块宽度的比例，默认为0.2

    返回:
    分割后的图片块tensor列表和每个块的位置信息 [(tensor, (start_x, end_x)), ...]
    """
    # 获取tensor的维度
    batch_size, channels, height, width = image_tensor.shape
    aspect_ratio = width / height

    # 如果长宽比小于等于6:1，不需要分割
    if aspect_ratio <= 6.0:
        return [(image_tensor, (0, width))]

    # 计算每个块的宽度
    block_width = int(height * target_ratio)

    # 计算重叠的像素数
    overlap_pixels = int(block_width * overlap_ratio)

    # 计算有效步长（块宽度减去重叠部分）
    stride = block_width - overlap_pixels

    # 计算需要的块数
    num_blocks = max(1, int((width - overlap_pixels) / stride) + 1)

    # 存储分割后的图片块和位置信息
    image_blocks_with_pos = []

    # 处理所有块
    for i in range(num_blocks):
        # 计算起始位置
        start_x = i * stride
        # 确保不超出原图宽度
        end_x = min(start_x + block_width, width)

        # 如果是最后一块且不足目标比例
        # if end_x == width and end_x - start_x < block_width:
        #     # 从右向左取足够的像素确保4:1比例
        #     start_x = max(0, width - block_width)

        # 提取当前块
        current_block = image_tensor[:, :, :, start_x:end_x]
        image_blocks_with_pos.append((current_block, (start_x, end_x)))

        # 如果已经到达图片右边界，停止分割
        if end_x == width:
            break

    return image_blocks_with_pos


def split_text_annotation(text, image_blocks_with_pos, original_width):
    """
    根据图片分割的位置信息拆分文字标注

    参数:
    text: 原始文字标注字符串
    image_blocks_with_pos: 图片分割后的块和位置信息列表 [(tensor, (start_x, end_x)), ...]
    original_width: 原始图片的宽度

    返回:
    分割后的文字标注列表
    """
    text_blocks = []
    text_length = len(text)

    for _, (start_x, end_x) in image_blocks_with_pos:
        # 计算当前块占原图的比例
        start_ratio = start_x / original_width
        end_ratio = end_x / original_width

        # 计算对应的文字起始和结束位置
        text_start = int(start_ratio * text_length)
        text_end = int(end_ratio * text_length)

        # 确保文字索引在有效范围内
        text_start = max(0, text_start)
        text_end = min(text_length, text_end)

        # 提取当前块对应的文字
        current_text_block = text[text_start:text_end]
        text_blocks.append(current_text_block)

    return text_blocks


def merge_image_blocks(
    processed_blocks_with_pos, original_shape, blend_method="weighted"
):
    """
    将处理后的图片块重新组合成完整图片，使用二维权重掩码进行融合

    参数:
    processed_blocks_with_pos: 处理后的图片块和位置信息列表 [(tensor, (start_x, end_x)), ...]
    original_shape: 原始图片的形状 (batch_size, channels, height, width)
    blend_method: 重叠区域的混合方法，可选 'weighted'(权重混合) 或 'max'(取最大值)

    返回:
    重组后的完整图片tensor
    """
    batch_size, channels, height, width = original_shape

    # 如果只有一个块，直接返回
    if len(processed_blocks_with_pos) == 1:
        return processed_blocks_with_pos[0][0]

    # 创建输出tensor
    output = torch.zeros(
        original_shape,
        dtype=processed_blocks_with_pos[0][0].dtype,
        device=processed_blocks_with_pos[0][0].device,
    )

    # 创建权重图，用于混合重叠区域
    weight_map = torch.zeros(
        (batch_size, 1, height, width),
        dtype=torch.float32,
        device=processed_blocks_with_pos[0][0].device,
    )

    # 遍历所有块进行合并
    for block, (start_x, end_x) in processed_blocks_with_pos:
        block_width = end_x - start_x

        if blend_method == "weighted":
            # 为每个块创建二维权重掩码
            # 水平方向权重
            w_weight = torch.linspace(0, 1, steps=block_width, device=block.device)
            w_weight = torch.minimum(w_weight, 1 - w_weight)  # 形成中间强、边缘弱的权重

            # 调整边缘块的权重
            if start_x == 0:  # 第一个块
                w_weight[: block_width // 2] = w_weight[block_width // 2]
            if end_x == width:  # 最后一个块
                w_weight[block_width // 2 :] = w_weight[block_width // 2]

            # 扩展权重到二维
            w_weight = w_weight.view(1, 1, 1, block_width)
            weight = w_weight.expand(batch_size, 1, height, block_width)

            # 扩展权重到所有通道
            expanded_weight = weight.expand_as(block)

            # 更新输出和权重图
            output[:, :, :, start_x:end_x] += block * expanded_weight
            weight_map[:, :, :, start_x:end_x] += weight

        elif blend_method == "max":
            # 取最大值方法
            current_output = output[:, :, :, start_x:end_x]
            if current_output.shape != block.shape:
                block = F.interpolate(block, current_output.shape[-2:],mode="bilinear")
            output[:, :, :, start_x:end_x] = torch.max(current_output, block)

    # 对于权重混合方法，需要归一化
    if blend_method == "weighted":
        # 避免除以零
        weight_map = torch.clamp(weight_map, min=1e-6)
        # 归一化输出
        output = output / weight_map.expand_as(output)

    return output

