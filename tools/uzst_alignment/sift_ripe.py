import cv2
import numpy as np
import os
import torch
from torchvision.utils import save_image
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

import kornia.feature as KF
import kornia.geometry as KG
from pathlib import Path
from torchvision.transforms import ToTensor, ToPILImage

from RIPE.ripe.utils.utils import cv2_matches_from_kornia, resize_image, to_cv_kpts
from RIPE.ripe import vgg_hyper

from utils.wavelet import adaptive_instance_normalization, wavelet_color_fix,wavelet_color_fix_cv2

def compute_and_save_residual(img1, img2, name1, name2, output_dir, img_name):
    # 确保输出目录存在
    residual_dir = os.path.join(output_dir, "residual")
    os.makedirs(residual_dir, exist_ok=True)
    
    # 彩色残差 (PyTorch方式)
    img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    diff_color = img1_tensor - img2_tensor
    diff_color = (diff_color - diff_color.min()) / (diff_color.max() - diff_color.min())
    save_image(diff_color, os.path.join(residual_dir, f'{img_name}_residual_{name1}_vs_{name2}_color.png'))
    
    # 计算平均绝对差异
    mean_abs_diff = np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32)))
    
    return mean_abs_diff

def align_image_ripe(src_img, tgt_img, threshold=0.7):
    # 检查设备
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载RIPE模型
    model = vgg_hyper(Path('RIPE/weights/ripe_weights.pth')).to(dev)
    model.eval()
    
    # 保存原始图像尺寸
    src_h, src_w = src_img.shape[:2]
    tgt_h, tgt_w = tgt_img.shape[:2]
    
    # 转换图像为torch张量并归一化
    # 假设src_img和tgt_img是OpenCV格式的图像(numpy数组)
    src_tensor = torch.from_numpy(src_img).float().permute(2, 0, 1).to(dev) / 255.0
    tgt_tensor = torch.from_numpy(tgt_img).float().permute(2, 0, 1).to(dev) / 255.0
    
    # 记录调整前的尺寸
    src_orig_h, src_orig_w = src_tensor.shape[1:3]
    tgt_orig_h, tgt_orig_w = tgt_tensor.shape[1:3]
    
    # 调整图像尺寸
    src_tensor = resize_image(src_tensor)
    tgt_tensor = resize_image(tgt_tensor)
    
    # 记录调整后的尺寸
    src_resized_h, src_resized_w = src_tensor.shape[1:3]
    tgt_resized_h, tgt_resized_w = tgt_tensor.shape[1:3]
    
    # 计算缩放因子
    src_scale_x_to_resized = src_resized_w / src_orig_w
    src_scale_y_to_resized = src_resized_h / src_orig_h
    tgt_scale_x_to_resized = tgt_resized_w / tgt_orig_w
    tgt_scale_y_to_resized = tgt_resized_h / tgt_orig_h
    
    # 检测关键点和描述符
    kpts_1, desc_1, score_1 = model.detectAndCompute(src_tensor, threshold=0.5, top_k=2048)
    kpts_2, desc_2, score_2 = model.detectAndCompute(tgt_tensor, threshold=0.5, top_k=2048)
    
    # 使用最近邻匹配器
    matcher = KF.DescriptorMatcher("mnn")
    try:
        match_dists, match_idxs = matcher(desc_1, desc_2)
    except Exception as e:
        print(f"Error during matching: {e}")
        return None
    # 获取匹配的关键点
    matched_pts_1 = kpts_1[match_idxs[:, 0]]
    matched_pts_2 = kpts_2[match_idxs[:, 1]]
    
    # 检查是否有足够的匹配点
    if len(matched_pts_1) < 4:
        print("匹配点不足，跳过对齐")
        return None
    
    # 使用RANSAC查找单应性矩阵
    try:
        pts1_np = matched_pts_1.cpu().numpy()
        pts2_np = matched_pts_2.cpu().numpy()
        
        # 使用OpenCV的USAC_MAGSAC方法
        H_resized_np, mask = cv2.findHomography(pts1_np, pts2_np, cv2.USAC_MAGSAC, 1.0)
        # 获取缩放后图像上的单应性矩阵
        # H_resized, mask = KG.ransac.RANSAC(model_type="homography", inl_th=1.0)(matched_pts_1, matched_pts_2)
        
        # 将H_resized转换为numpy数组以便进行矩阵操作
        # H_resized_np = H_resized.cpu().numpy()
        # print(H_resized_np.shape)
        # src_img = src_tensor.cpu().numpy().transpose(1, 2, 0) * 255.0
        # tgt_img = tgt_tensor.cpu().numpy().transpose(1, 2, 0) * 255.0
        # warped_src_img = cv2.warpPerspective(src_img.astype(np.uint8), H_resized_np,
        #                                      (tgt_resized_w, tgt_resized_h))
        # cv2.imwrite("warped_src_img.png", warped_src_img)
        # cv2.imwrite("tgt_img.png", tgt_img.astype(np.uint8))
        # print("Checking warped:")
        # 创建源图像和目标图像的缩放矩阵
        S_src = np.array([
            [src_scale_x_to_resized, 0, 0],
            [0, src_scale_y_to_resized, 0],
            [0, 0, 1]
        ])
        # print(S_src)
        S_tgt_inv = np.array([
            [1.0/tgt_scale_x_to_resized, 0, 0],
            [0, 1.0/tgt_scale_y_to_resized, 0],
            [0, 0, 1]
        ])
        # print(S_tgt_inv)
        # 计算原始尺寸上的单应性矩阵: H = S_tgt_inv * H_resized * S_src
        H = S_tgt_inv @ H_resized_np @ S_src
        # print(H)
        return H
    except Exception as e:
        print(f"Error finding homography: {e}")
        return None

def align_image(src_img, tgt_img, threshold=0.7):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2)
    
    kp1, des1 = sift.detectAndCompute(src_img, None)
    kp2, des2 = sift.detectAndCompute(tgt_img, None)

    try:
        matches = bf.knnMatch(des1, des2, k=2)
    except cv2.error as e:
        print(f"OpenCV error during matching: {e}")
        return None
    good = [m for m, n in matches if m.distance < threshold * n.distance]

    if len(good) < 4:
        print("匹配点不足，跳过对齐")
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(pts1, pts2, cv2.USAC_MAGSAC, 5.0)
    return H

def resize_to_max_edge(image, max_edge_length=1024):
    """将图像等比例缩放，使得最长边为指定长度"""
    height, width = image.shape[:2]
    
    # 计算缩放比例
    scale = max_edge_length / max(height, width)
    
    # 计算新的尺寸
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # 缩放图像
    resized_image = cv2.resize(image, (new_width, new_height))
    
    return resized_image, scale

def create_scale_matrix(scale, inverse=False):
    scale_value = 1.0/scale if inverse else scale
    return np.array([
        [scale_value, 0, 0],
        [0, scale_value, 0],
        [0, 0, 1]
    ])

# 替换所有的cv2.imwrite调用
def imwrite_unicode(filename, img):
    """保存图像，支持中文路径"""
    ret, buf = cv2.imencode('.png', img)
    if ret:
        with open(filename, 'wb') as f:
            f.write(buf)
        return True
    return False

def perform_cascade_alignment(source_img, target_img, intermediate_imgs=None, name_prefix="", 
                             img_name="", save_dir="", cascade_alignment_dir="",Fine_thresh=0.3):
    """
    执行级联图像对齐
    
    参数:
    - source_img: 源图像
    - target_img: 目标图像
    - intermediate_imgs: 中间图像列表 [img1, img2, ...] (可选)
    - name_prefix: 源图像名称前缀，如 "14mm"
    - img_name: 图像基本名称
    - save_dir: 保存差异图像的目录
    - cascade_alignment_dir: 保存对齐结果的目录
    
    返回:
    - aligned_img: 对齐后的图像
    - avg_diff: 平均绝对差异
    """
    print(f"\n执行级联对齐: {name_prefix} -> {'' if not intermediate_imgs else ' -> '.join([f'中间{i+1}' for i in range(len(intermediate_imgs))]) + ' -> '}目标图像...")
    
    # 如果没有中间图像，直接对齐到目标
    if not intermediate_imgs:
        # 初始对齐
        H = align_image(source_img, target_img)
        aligned_img = cv2.warpPerspective(source_img, H, (target_img.shape[1], target_img.shape[0]))
        
        # 精细对齐
        H_refined = align_image(aligned_img, target_img, threshold=0.3)
        if H_refined is None:
            H_refined = align_image(aligned_img, target_img, threshold=0.75)
        if H_refined is None:
            return None,255
        H_cascade = H_refined @ H
        
        # 最终对齐
        final_aligned_img = cv2.warpPerspective(source_img, H_cascade, 
                                              (target_img.shape[1], target_img.shape[0]))
    else:
        # 级联对齐
        current_img = source_img
        # transforms = []
        
        # 1. 对齐到所有中间图像
        for i, intermediate_img in enumerate(intermediate_imgs):
            H = align_image(current_img, intermediate_img)
            if H is None:
                return None,255
            # transforms.append(H)
            current_img = cv2.warpPerspective(current_img, H, 
                                            (intermediate_img.shape[1], intermediate_img.shape[0]))
        
        # 2. 对齐到目标图像
        prev_img = current_img
        H_to_target = align_image(current_img, target_img)
        if H_to_target is None:
            return None,255
        # transforms.append(H_to_target)
        current_img = cv2.warpPerspective(current_img, H_to_target, 
                                        (target_img.shape[1], target_img.shape[0]))
        
        # 3. 精细对齐
        # H_refined = align_image(current_img, target_img, threshold=Fine_thresh)
        H_refined = align_image_ripe(current_img, target_img, threshold=Fine_thresh)
        if H is None:
            return None,255
        # transforms.append(H_refined)
        if H_refined is None:
            return None,255
        # final_aligned_img = cv2.warpPerspective(current_img, H_refined,
        #                                       (target_img.shape[1], target_img.shape[0]))
        # 4. 计算级联变换矩阵
        H_cascade = H_refined @ H_to_target
        
        # 5. 应用级联变换矩阵进行最终对齐
        final_aligned_img = cv2.warpPerspective(prev_img, H_cascade, 
                                              (target_img.shape[1], target_img.shape[0]))
        
    
    # 保存结果
    if cascade_alignment_dir and img_name:
        final_aligned_img_pil = wavelet_color_fix(cv2.cvtColor(final_aligned_img, cv2.COLOR_BGR2RGB),cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
        final_aligned_img_pil.save(os.path.join(cascade_alignment_dir, f"{img_name}_{name_prefix}.png"))
        imwrite_unicode(os.path.join(cascade_alignment_dir,"wocolorfix", f"{img_name}_{name_prefix}.png"), final_aligned_img)
    
    # 计算并保存差异
    avg_diff = 0
    if save_dir and img_name:
        avg_diff = compute_and_save_residual(final_aligned_img, target_img, 
                                          f"{name_prefix}_cascade", "target", save_dir, img_name)
        
        path_description = f"{name_prefix}"
        if intermediate_imgs:
            for i in range(len(intermediate_imgs)):
                path_description += f" -> 中间{i+1}"
        path_description += " -> 目标"
        
        print(f"级联对齐结果: {path_description}, 平均绝对差异 = {avg_diff:.2f}")
    
    return final_aligned_img, avg_diff

# 主程序
def main():
    # 创建基础输出目录
    save_dir = '/data/vjuicefs_ai_camera_llm/intern_data/minxing/datasets/Selected lr/ForPaper2/'
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建子目录
    original_dir = os.path.join(save_dir, "original")
    cascade_alignment_dir = os.path.join(save_dir, "cascade_alignment")
    residual_dir = os.path.join(save_dir, "residual")
    
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(cascade_alignment_dir, exist_ok=True)
    os.makedirs(os.path.join(cascade_alignment_dir,"wocolorfix"), exist_ok=True)
    os.makedirs(residual_dir, exist_ok=True)
    
    # 图像读取
    # root_dir = 'TestSet'
    root_dir = '/data/vjuicefs_ai_camera_llm/intern_data/minxing/datasets/Selected lr/ForPaper/original'
    all_files = os.listdir(root_dir)
    img_names = [file.replace("_200mm_cr.png", "") for file in all_files if file.endswith("_200mm_cr.png")]
    img_names = sorted(set(img_names))  # 去重并排序
    
    for img_name in img_names:
        if os.path.exists(os.path.join(cascade_alignment_dir, f"{img_name}_14mm.png")):
            print(f"{img_name} already exists! Skipping...")
            continue
        print(img_name)
        print(os.path.join(root_dir, f'{img_name}_200mm.png'))
        
        img_200mm_cr = cv2.imdecode(np.fromfile(os.path.join(root_dir, f'{img_name}_200mm_cr.png'), dtype=np.uint8), cv2.IMREAD_COLOR)
        img_200mm = cv2.imdecode(np.fromfile(os.path.join(root_dir, f'{img_name}_200mm.png'), dtype=np.uint8), cv2.IMREAD_COLOR)
        # height, width = img_200mm.shape[:2]
        # # 计算旋转矩阵
        # rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 180, 1)
        # img_200mm = cv2.warpAffine(img_200mm, rotation_matrix, (width, height))

        img_85mm = cv2.imdecode(np.fromfile(os.path.join(root_dir, f'{img_name}_85mm.png'), dtype=np.uint8), cv2.IMREAD_COLOR)
        img_35mm = cv2.imdecode(np.fromfile(os.path.join(root_dir, f'{img_name}_35mm.png'), dtype=np.uint8), cv2.IMREAD_COLOR)
        img_14mm = cv2.imdecode(np.fromfile(os.path.join(root_dir, f'{img_name}_14mm.png'), dtype=np.uint8), cv2.IMREAD_COLOR)

        # 缩小图像选ROI
        # scale = 0.15
        # resized_img = cv2.resize(img_200mm, (0, 0), fx=scale, fy=scale)
        # roi = cv2.selectROI("Select ROI (based on 200mm image)", resized_img, fromCenter=False, showCrosshair=True)
        # cv2.destroyAllWindows()
        # # ROI 映射回原图
        # x, y, w, h = [int(v / scale) for v in roi]
        # img_200mm = img_200mm[y:y+h, x:x+w]
        
        # 保存原始图像
        # imwrite_unicode(os.path.join(original_dir, f"{img_name}_200mm_cr.png"), img_200mm_cr)
        # imwrite_unicode(os.path.join(original_dir, f"{img_name}_200mm.png"), img_200mm)
        # imwrite_unicode(os.path.join(original_dir, f"{img_name}_85mm.png"), img_85mm)
        # imwrite_unicode(os.path.join(original_dir, f"{img_name}_35mm.png"), img_35mm)
        # imwrite_unicode(os.path.join(original_dir, f"{img_name}_14mm.png"), img_14mm)

        imwrite_unicode(os.path.join(cascade_alignment_dir, f"{img_name}_200mm.png"), img_200mm_cr)

        # 执行两阶段对齐
        print("\n开始两阶段图像对齐...")
        if not os.path.exists(os.path.join(cascade_alignment_dir, f"{img_name}_85mm.png")):
            # 85mm -> 200mm (直接对齐)
            aligned_85mm, diff_85mm = perform_cascade_alignment(
                source_img=img_85mm, 
                target_img=img_200mm_cr, 
                intermediate_imgs=[img_200mm],
                name_prefix="85mm", 
                img_name=img_name, 
                save_dir=save_dir, 
                cascade_alignment_dir=cascade_alignment_dir,
                Fine_thresh=0.3
            )
            print("\n对齐结果比较:")
            print(f"85mm -> 200mm: 平均绝对差异 = {diff_85mm:.2f}")
        if not os.path.exists(os.path.join(cascade_alignment_dir, f"{img_name}_35mm.png")):
            # 35mm -> 85mm -> 200mm (级联对齐)
            aligned_35mm, diff_35mm = perform_cascade_alignment(
                source_img=img_35mm, 
                target_img=img_200mm_cr, 
                intermediate_imgs=[img_85mm, img_200mm], 
                name_prefix="35mm", 
                img_name=img_name, 
                save_dir=save_dir, 
                cascade_alignment_dir=cascade_alignment_dir,
                Fine_thresh=0.3
            )
            print("\n对齐结果比较:")
            print(f"35mm -> 85mm -> 200mm: 平均绝对差异 = {diff_35mm:.2f}")
        if not os.path.exists(os.path.join(cascade_alignment_dir, f"{img_name}_14mm.png")):
            # 14mm -> 35mm -> 85mm -> 200mm (级联对齐)
            aligned_14mm, diff_14mm = perform_cascade_alignment(
                source_img=img_14mm, 
                target_img=img_200mm_cr, 
                intermediate_imgs=[img_35mm, img_85mm, img_200mm], 
                name_prefix="14mm", 
                img_name=img_name, 
                save_dir=save_dir, 
                cascade_alignment_dir=cascade_alignment_dir,
                Fine_thresh=0.7
            )
            print(f"14mm -> 35mm -> 85mm -> 200mm: 平均绝对差异 = {diff_14mm:.2f}")
        
        # 打印对齐结果比较
    

        print(f"\n图像 {img_name} 对齐完成，结果保存在: {save_dir}")

if __name__ == "__main__":
    main()
