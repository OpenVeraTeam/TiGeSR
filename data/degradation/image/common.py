import math
import random

import cv2
import numpy as np
from numpy.core.multiarray import array
import torch
from PIL import Image
from torch.nn import functional as F


def min_edge_crop(image, image_size):
    h, w = image.shape[0], image.shape[1]
    if w < h:
        overpad = random.randint(0, round((h - w) + 1))
        new_arr = image[overpad:overpad + w, :]
        pil_image = Image.fromarray(new_arr)
        new_arr = np.array(pil_image.resize((image_size, image_size), Image.LANCZOS))
        return new_arr
    else:
        overpad = random.randint(0, round((w - h) + 1))
        new_arr = image[:, overpad:overpad + h]
        pil_image = Image.fromarray(new_arr)
        new_arr = np.array(pil_image.resize((image_size, image_size), Image.LANCZOS))
        return new_arr


def size_scope_crop(image, image_size):
    h, w = image.shape[0], image.shape[1]

    increment_scale = int(image_size / 2)
    scale_max = int((min(h, w) - image_size) / increment_scale)
    # if scale_max == 0:
    #     new_image_size = int(image_size)
    # else:
    new_image_size = int(image_size + np.random.randint(0, scale_max + 1) * increment_scale)

    if np.random.random() < 0.3:
        overpad_w = random.randint(0, round((w - new_image_size) + 1))
        overpad_h = random.randint(0, round((h - new_image_size) + 1))
        new_arr = image[overpad_h:overpad_h + new_image_size, overpad_w:overpad_w + new_image_size, :]
    else:
        new_arr = image[round((h - new_image_size) / 2):round((h - new_image_size) / 2) +
                        new_image_size, round((w - new_image_size) / 2):round((w - new_image_size) / 2) + new_image_size, :]

    pil_image = Image.fromarray(new_arr)
    new_arr = np.array(pil_image.resize((image_size, image_size), Image.LANCZOS))
    return new_arr


def multi_scale_crop(pil_image, image_size):
    arr = np.array(pil_image).copy()
    h, w = arr.shape[0], arr.shape[1]
    pro = np.random.random()

    # 最小边小于image_size，翻转pad
    if min(h, w) < image_size:
        new_arr = np.zeros((image_size, image_size, 3))
        if w > h:
            if w > image_size:
                overpad_w = random.randint(0, round((w - image_size) + 1))
                new_arr[0:h, :, :] = arr[:, overpad_w:overpad_w + image_size, :]
                arr_flip = arr[::-1, :, :]
                new_arr[h:image_size, :, :] = arr_flip[0:image_size - h, overpad_w:overpad_w + image_size, :]
            else:
                pil_image_new = pil_image.resize((image_size, image_size), Image.LANCZOS)
                new_arr = np.array(pil_image_new)
        else:
            if h > image_size:
                overpad_h = random.randint(0, round((h - image_size) + 1))
                new_arr[:, 0:w, :] = arr[overpad_h:overpad_h + image_size, :, :]
                arr_flip = arr[:, ::-1, :]
                new_arr[:, w:image_size, :] = arr_flip[overpad_h:overpad_h + image_size, 0:image_size - w, :]
            else:
                pil_image_new = pil_image.resize((image_size, image_size), Image.LANCZOS)
                new_arr = np.array(pil_image_new)
        return new_arr
    elif h == image_size and w == image_size:
        return arr
    else:
        if pro < 0.1:
            return min_edge_crop(arr, image_size)
        elif 0.1 < pro < 0.6:
            return size_scope_crop(arr, image_size)
        elif 0.6 < pro < 0.8:
            return center_crop_arr(pil_image, image_size)
        else:
            return random_crop_arr(pil_image, image_size)


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2

    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py
def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)

    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/data/transforms.py
def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/img_process_util.py
def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


# https://github.com/XPixelGroup/BasicSR/blob/033cd6896d898fdd3dcda32e3102a792efa1b8f4/basicsr/utils/color_util.py#L186
def rgb2ycbcr_pt(img, y_only=False):
    """Convert RGB images to YCbCr images (PyTorch version).

    It implements the ITU-R BT.601 conversion for standard-definition television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    Args:
        img (Tensor): Images with shape (n, 3, h, w), the range [0, 1], float, RGB format.
         y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        (Tensor): converted images with the shape (n, 3/1, h, w), the range [0, 1], float.
    """
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    out_img = out_img / 255.
    return out_img


def to_pil_image(inputs, mem_order, val_range, channel_order):
    # convert inputs to numpy array
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.cpu().numpy()
    assert isinstance(inputs, np.ndarray)

    # make sure that inputs is a 4-dimension array
    if mem_order in ["hwc", "chw"]:
        inputs = inputs[None, ...]
        mem_order = f"n{mem_order}"
    # to NHWC
    if mem_order == "nchw":
        inputs = inputs.transpose(0, 2, 3, 1)
    # to RGB
    if channel_order == "bgr":
        inputs = inputs[..., ::-1].copy()
    else:
        assert channel_order == "rgb"

    if val_range == "0,1":
        inputs = inputs * 255
    elif val_range == "-1,1":
        inputs = (inputs + 1) * 127.5
    else:
        assert val_range == "0,255"

    inputs = inputs.clip(0, 255).astype(np.uint8)
    return [inputs[i] for i in range(len(inputs))]


def put_text(pil_img_arr, text):
    cv_img = pil_img_arr[..., ::-1].copy()
    cv2.putText(cv_img, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return cv_img[..., ::-1].copy()


def auto_resize(img: Image.Image, size: int) -> Image.Image:
    short_edge = min(img.size)
    if short_edge < size:
        r = size / short_edge
        img = img.resize(
            tuple(math.ceil(x * r) for x in img.size), Image.BICUBIC
        )
    else:
        # make a deep copy of this image for safety
        img = img.copy()
    return img


def pad(img: np.ndarray, scale: int) -> np.ndarray:
    h, w = img.shape[:2]
    ph = 0 if h % scale == 0 else math.ceil(h / scale) * scale - h
    pw = 0 if w % scale == 0 else math.ceil(w / scale) * scale - w
    return np.pad(
        img, pad_width=((0, ph), (0, pw), (0, 0)), mode="constant",
        constant_values=0
    )


def GaussianBlur(img):
    k_size = 5
    sigma = np.random.choice(np.arange(0.8, 1.6, step=0.1))
    return cv2.GaussianBlur(img, (k_size, k_size), sigma)


def resize_degradation(img):
    H, W, _ = img.shape
    img_down_sampled = cv2.resize(img, dsize=(W // 2, H // 2), interpolation=cv2.INTER_CUBIC)
    return cv2.resize(img_down_sampled, dsize=(W, H), interpolation=cv2.INTER_CUBIC)


def JEPG_degradation(img):
    # 将图像编码为 JPEG 格式
    quality_parameter = random.randint(30, 70)  # JPEG 压缩的质量参数，范围为 0 到 100
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality_parameter]
    _, encimg = cv2.imencode('.jpg', img, encode_param)

    # 将编码后的图像解码
    decimg = cv2.imdecode(encimg, 1)
    return decimg
