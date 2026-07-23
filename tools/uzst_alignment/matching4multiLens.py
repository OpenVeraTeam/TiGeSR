import cv2
import numpy as np
import os
import torch
from torchvision.utils import save_image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile

from transformers import AutoImageProcessor, AutoModel, SuperPointForKeypointDetection
import torch
from PIL import Image
from kornia.feature import LoFTR
import requests

def visualize_matches(src_img, tgt_img, src_pts, tgt_pts, matches_mask=None, save_path=None):
    """可视化两张图像之间的匹配点"""
    h1, w1 = src_img.shape[:2]
    h2, w2 = tgt_img.shape[:2]
    
    # 创建拼接图像
    vis_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis_img[:h1, :w1] = src_img if len(src_img.shape) == 3 else cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
    vis_img[:h2, w1:w1+w2] = tgt_img if len(tgt_img.shape) == 3 else cv2.cvtColor(tgt_img, cv2.COLOR_GRAY2BGR)
    
    # 绘制匹配线
    if matches_mask is None:
        matches_mask = [1] * len(src_pts)
    
    # 计算有效匹配数
    valid_matches = sum(matches_mask)
    print(f"绘制 {valid_matches} 个有效匹配点")
    
    # 随机选择颜色
    np.random.seed(42)  # 固定随机种子以便复现
    colors = np.random.randint(0, 255, (len(src_pts), 3))
    
    for i, (src_pt, tgt_pt, mask) in enumerate(zip(src_pts, tgt_pts, matches_mask)):
        if mask:
            pt1 = (int(src_pt[0]), int(src_pt[1]))
            pt2 = (int(tgt_pt[0]) + w1, int(tgt_pt[1]))
            color = (int(colors[i][0]), int(colors[i][1]), int(colors[i][2]))
            cv2.line(vis_img, pt1, pt2, color, 1)
            cv2.circle(vis_img, pt1, 3, (0, 0, 255), -1)
            cv2.circle(vis_img, pt2, 3, (0, 0, 255), -1)
    
    # 添加标题和匹配点数量信息
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis_img, f"Matches: {valid_matches}", (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    if save_path:
        cv2.imwrite(save_path, vis_img)
        print(f"匹配可视化已保存到: {save_path}")
    
    return vis_img

def align_with_LoFTR(src_img, tgt_img, output_dir=None):
    from torchvision import transforms
    to_tensor = transforms.ToTensor()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input = {"image0": to_tensor(cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)).unsqueeze(0), "image1": to_tensor(cv2.cvtColor(tgt_img, cv2.COLOR_BGR2GRAY)).unsqueeze(0)}
    print(to_tensor(cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)).unsqueeze(0).shape)
    loftr = LoFTR('outdoor').to(device)
    # 将输入移动到相同设备
    for k in input:
        if isinstance(input[k], torch.Tensor):
            input[k] = input[k].to(device)
    with torch.no_grad():
        out = loftr(input)
    keypoints0 = out['keypoints0'].cpu().numpy()
    keypoints1 = out['keypoints1'].cpu().numpy()
    confidence = out['confidence'].cpu().numpy()
    # 筛选高置信度的匹配点
    conf_threshold = 0.6
    valid_matches = confidence > conf_threshold
    
    if np.sum(valid_matches) < 4:
        print(f"找到的有效匹配点不足 ({np.sum(valid_matches)} < 4)，无法计算单应性矩阵")
        return None
    
    # 提取有效的关键点对
    src_pts = keypoints0[valid_matches]
    tgt_pts = keypoints1[valid_matches]
    
    print(f"LoFTR找到 {len(keypoints0)} 个匹配点，其中 {np.sum(valid_matches)} 个有效匹配点")
    
    # 使用RANSAC计算单应性矩阵
    H, mask = cv2.findHomography(src_pts, tgt_pts, cv2.RANSAC, 5.0)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        vis_img = visualize_matches(
            src_img, tgt_img,
            src_pts, tgt_pts,
            mask.ravel().tolist(),
            os.path.join(output_dir, "loftr_matches.jpg")
        )
    return H

def compute_and_save_residual(img1, img2, name1, name2, output_dir):
    # 确保输出目录存在
    residual_dir = os.path.join(output_dir, "residual")
    os.makedirs(residual_dir, exist_ok=True)
    
    # 灰度残差
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    diff_gray = cv2.absdiff(gray1, gray2)
    norm_diff_gray = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(residual_dir, f'residual_{name1}_vs_{name2}_gray.jpg'), norm_diff_gray)
    
    # 彩色残差 (PyTorch方式)
    img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    diff_color = img1_tensor - img2_tensor
    diff_color = (diff_color - diff_color.min()) / (diff_color.max() - diff_color.min())
    save_image(diff_color, os.path.join(residual_dir, f'residual_{name1}_vs_{name2}_color.jpg'))
    
    # 计算平均绝对差异
    mean_abs_diff = np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32)))
    
    return mean_abs_diff

# 使用SIFT进行特征提取和匹配
def align_image(src_img, tgt_img):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2)
    
    kp1, des1 = sift.detectAndCompute(src_img, None)
    kp2, des2 = sift.detectAndCompute(tgt_img, None)

    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 4:
        print("匹配点不足，跳过对齐")
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
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

def main():
    # 路径设置
    root_dir = "/data/vjuicefs_ai_camera_llm/intern_data/minxing/datasets/V200Ultra_Collected"
    img_200mm_path = os.path.join(root_dir, '灭火器_200mm.jpg')
    img_85mm_path = os.path.join(root_dir,  '灭火器_85mm.jpg')

    # 图像读取
    img_200mm = cv2.imread(img_200mm_path) # H,W,C
    if img_200mm is None:
        print(f"无法读取图像: {img_200mm_path}")
        return
        
    height, width = img_200mm.shape[:2]
    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 180, 1)
    img_200mm = cv2.warpAffine(img_200mm, rotation_matrix, (width, height))
    
    img_85mm = cv2.imread(img_85mm_path)
    if img_85mm is None:
        print(f"无法读取图像: {img_85mm_path}")
        return

    # 创建输出目录
    save_dir = './output_sg_compare'
    residual_dir = os.path.join(save_dir, "residual")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(residual_dir, exist_ok=True)

    # 保存原始图像副本
    img_85mm_orig = img_85mm.copy()
    img_200mm_orig = img_200mm.copy()
    
    # 将图像缩放到长边为1024
    print("\n将图像缩放到长边为1024...")
    img_85mm_resized, scale_85mm = resize_to_max_edge(img_85mm, 1024)
    img_200mm_resized, scale_200mm = resize_to_max_edge(img_200mm, 1024)
    
    print(f"85mm图像缩放比例: {scale_85mm:.4f}")
    print(f"200mm图像缩放比例: {scale_200mm:.4f}")
    scale_matrix_src = np.array([
        [scale_85mm, 0, 0],
        [0, scale_85mm, 0],
        [0, 0, 1]
    ])
    
    scale_matrix_dst = np.array([
        [1.0/scale_200mm, 0, 0],
        [0, 1.0/scale_200mm, 0],
        [0, 0, 1]
    ])
    
    # 保存缩放后的图像用于参考
    cv2.imwrite(os.path.join(save_dir, '85mm_resized.jpg'), img_85mm_resized)
    cv2.imwrite(os.path.join(save_dir, '200mm_resized.jpg'), img_200mm_resized)

    # 第一阶段：使用SIFT方法对齐缩放后的图像
    print("\n第一阶段：使用SIFT方法对齐...")
    H_sift = align_image(img_85mm_resized, img_200mm_resized)
    
    if H_sift is None:
        print("SIFT方法失败，无法继续进行")
        return
    
    # 应用SIFT变换到缩放后的图像
    sift_transformed = cv2.warpPerspective(img_85mm_orig, scale_matrix_dst @ H_sift @ scale_matrix_src, 
                                                  (img_200mm_orig.shape[1], img_200mm_orig.shape[0]))
    
    # 保存第一阶段结果
    cv2.imwrite(os.path.join(save_dir, 'stage1_sift_result.jpg'), sift_transformed)
    sift_transformed_resized, scale_sift = resize_to_max_edge(sift_transformed, 1024)
    # 计算与目标图像的差异
    mean_diff_sift = compute_and_save_residual(sift_transformed_resized, img_200mm_resized, 
                                              "stage1_sift", "200mm_resized", save_dir)
    print(f"第一阶段 SIFT方法: 平均绝对差异 = {mean_diff_sift:.2f}")
    
    # 第二阶段：使用SuperPoint+SuperGlue进一步优化
    # print("\n第二阶段：使用SuperPoint+SuperGlue进一步优化...")
    H_sp = align_with_LoFTR(sift_transformed_resized, img_200mm_resized,output_dir=save_dir)
    # H_sp = None
    
    if H_sp is None:
        print("SuperPoint+SuperGlue方法失败，使用SIFT结果作为最终结果")
        # 使用SIFT结果作为最终结果
        final_H_resized = H_sift
    else:
        # 组合两个变换矩阵
        final_H_resized = H_sp @ H_sift
        
        # 应用组合变换到缩放后的图像
        sp_transformed_resized = cv2.warpPerspective(img_85mm_resized, final_H_resized, 
                                                   (img_200mm_resized.shape[1], img_200mm_resized.shape[0]))
        
        # 保存第二阶段结果
        cv2.imwrite(os.path.join(save_dir, 'stage2_superpoint_result.jpg'), sp_transformed_resized)
        
        # 计算与目标图像的差异
        mean_diff_sp = compute_and_save_residual(sp_transformed_resized, img_200mm_resized, 
                                               "stage2_superpoint", "200mm_resized", save_dir)
        print(f"第二阶段 SuperPoint+SuperGlue方法: 平均绝对差异 = {mean_diff_sp:.2f}")
    
    # 调整变换矩阵以应用于原始尺寸图像
    # 创建尺度调整矩阵
    print(final_H_resized)
    # 计算适用于原始尺寸图像的变换矩阵
    final_H = scale_matrix_dst @ final_H_resized @ scale_matrix_src
    print(final_H)
    
    # 应用最终变换到原始尺寸图像
    final_transformed = cv2.warpPerspective(img_85mm_orig, final_H, 
                                          (img_200mm_orig.shape[1], img_200mm_orig.shape[0]))
    
    # 保存最终结果
    cv2.imwrite(os.path.join(save_dir, 'final_result.jpg'), final_transformed)
    
    # 计算最终结果与目标图像的差异
    mean_diff_final = compute_and_save_residual(final_transformed, img_200mm_orig, 
                                              "final", "200mm_orig", save_dir)
    print(f"最终结果: 平均绝对差异 = {mean_diff_final:.2f}")
    
    # 创建一个包含所有结果的对比图
    images_to_show = [
        img_85mm_orig, img_200mm_orig, 
        cv2.resize(img_85mm_resized, (img_85mm_orig.shape[1]//2, img_85mm_orig.shape[0]//2)),
        cv2.resize(img_200mm_resized, (img_200mm_orig.shape[1]//2, img_200mm_orig.shape[0]//2)),
        final_transformed
    ]
    
    titles = [
        "原始85mm图像", "目标200mm图像", 
        "缩放后85mm图像", "缩放后200mm图像", 
        "最终对齐结果"
    ]
    
    # 计算拼接图的行列数
    n_images = len(images_to_show)
    n_cols = min(3, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # 创建拼接图
    h, w = img_85mm_orig.shape[:2]
    comparison = np.zeros((h * n_rows, w * n_cols, 3), dtype=np.uint8)
    
    for i, (img, title) in enumerate(zip(images_to_show, titles)):
        row = i // n_cols
        col = i % n_cols
        
        # 确保图像尺寸一致
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
            
        comparison[row*h:(row+1)*h, col*w:(col+1)*w] = img
        
        # 添加标题
        cv2.putText(comparison, title, (col*w + 10, row*h + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 保存对比图
    cv2.imwrite(os.path.join(save_dir, 'comparison.jpg'), comparison)
    print(f"已保存所有方法的对比图到 {os.path.join(save_dir, 'comparison.jpg')}")


if __name__ == "__main__":
    main()
