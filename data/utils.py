import cv2
import os
import torch
from typing import List
from skimage.transform._geometric import _umeyama as get_sym_mat
import numpy as np
import torch.nn.functional as F

def adjust_image_cv2(box, img):
    # img: (C,H,W), box: 4x2 in img坐标（顺序 tl,tr,br,bl 已确认）
    C, H, W = img.shape
    pts1 = np.float32([box[0], box[1], box[2], box[3]])
    # 估计输出矩形的目标尺寸（像素个数，整数）
    edge_top    = np.linalg.norm(pts1[0] - pts1[1])
    edge_bottom = np.linalg.norm(pts1[3] - pts1[2])
    edge_left   = np.linalg.norm(pts1[0] - pts1[3])
    edge_right  = np.linalg.norm(pts1[1] - pts1[2])
    
    out_W = int(round(max(edge_top, edge_bottom)))
    out_H = int(round(max(edge_left, edge_right)))
    
    # 避免0
    out_W = max(out_W, 1)
    out_H = max(out_H, 1)
    
    # 像素中心模型：目标矩形的四点坐标用 [0,0],[out_W-1,0],[out_W-1,out_H-1],[0,out_H-1]
    dst = np.float32([[0, 0],
                      [out_W - 1, 0],
                      [out_W - 1, out_H - 1],
                      [0, out_H - 1]])
    
    # 用OpenCV算单应（更稳），得到 src(img)->dst(rect) 的 M
    M = cv2.getPerspectiveTransform(pts1.astype(np.float32), dst.astype(np.float32))
    
    # 直接透视裁剪到 out_H x out_W
    # 注意 dsize 顺序 (width, height)
    np_img = img.permute(1,2,0).contiguous().detach().cpu().numpy()
    cropped = cv2.warpPerspective(np_img, M, (out_W, out_H),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    out = torch.from_numpy(cropped)
    if out.dim()==2:
        out = out.unsqueeze(-1)
    out = out.permute(2,0,1).to(img.device, dtype=img.dtype)
    return out

def adjust_image(box, img):
    pts1 = np.float32([box[0], box[1], box[2], box[3]])
    width = max(np.linalg.norm(pts1[0]-pts1[1]), np.linalg.norm(pts1[2]-pts1[3]))
    height = max(np.linalg.norm(pts1[0]-pts1[3]), np.linalg.norm(pts1[1]-pts1[2]))
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    # get transform matrix
    M = get_sym_mat(pts1, pts2, estimate_scale=True)
    C, H, W = img.shape
    T = np.array([[2 / W, 0, -1], [0, 2 / H, -1], [0, 0, 1]])
    theta = np.linalg.inv(T @ M @ np.linalg.inv(T))
    theta = torch.from_numpy(theta[:2, :]).unsqueeze(0).type(torch.float32).to(img.device)
    grid = F.affine_grid(theta, torch.Size([1, C, H, W]), align_corners=True)
    result = F.grid_sample(img.unsqueeze(0), grid, align_corners=True)
    result = torch.clamp(result.squeeze(0), 0, 255)
    # crop
    result = result[:, :int(height), :int(width)]
    return result

def find_files_directory(
    directory,
    file_extensions=[
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
    ],
):
    """遍历目录及其子目录，查找所有图片并保存其路径"""
    files_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in file_extensions):
                files_paths.append(os.path.join(root, file))
    return files_paths


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == "float64":
                img = img.astype("float32")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)