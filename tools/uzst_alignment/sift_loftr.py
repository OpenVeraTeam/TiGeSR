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

# 计算残差
def compute_residual(img1, img2):
    # 彩色残差 (PyTorch方式)
    img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    diff_color = img1_tensor - img2_tensor
    diff_color = (diff_color - diff_color.min()) / (diff_color.max() - diff_color.min())
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
    diff = cv2.absdiff(gray1, gray2)
    norm_diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    mean_diff = np.mean(diff)
    return norm_diff, mean_diff,diff_color

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

def two_stage_alignment(src_img, tgt_img, output_dir=None, name_prefix="alignment"):
    """
    使用SIFT和LoFTR进行两阶段图像对齐
    
    Args:
        src_img: 源图像（需要对齐的图像）
        tgt_img: 目标图像（参考图像）
        output_dir: 输出目录，用于保存中间结果和可视化
        name_prefix: 输出文件名前缀
        
    Returns:
        final_H: 最终的变换矩阵
        aligned_img: 对齐后的图像
    """
    import cv2
    import numpy as np
    import os
    import torch
    from torchvision import transforms
    from kornia.feature import LoFTR
    
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "stage1"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "stage2"), exist_ok=True)
    
    # 保存原始图像副本
    src_orig = src_img.copy()
    tgt_orig = tgt_img.copy()
    
    # 将图像缩放到长边为1024以提高处理效率
    print("\n第一步：将图像缩放到适合处理的尺寸...")
    
    def resize_to_max_edge(image, max_edge_length=1024):
        height, width = image.shape[:2]
        scale = max_edge_length / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_image = cv2.resize(image, (new_width, new_height))
        return resized_image, scale
    
    src_resized, src_scale = resize_to_max_edge(src_img, 1024)
    tgt_resized, tgt_scale = resize_to_max_edge(tgt_img, 1024)
    
    print(f"源图像缩放比例: {src_scale:.4f}")
    print(f"目标图像缩放比例: {tgt_scale:.4f}")
    
    # 创建尺度变换矩阵
    scale_matrix_src = np.array([
        [src_scale, 0, 0],
        [0, src_scale, 0],
        [0, 0, 1]
    ])
    
    scale_matrix_dst = np.array([
        [1.0/tgt_scale, 0, 0],
        [0, 1.0/tgt_scale, 0],
        [0, 0, 1]
    ])
    
    # 保存缩放后的图像
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, f'{name_prefix}_src_resized.jpg'), src_resized)
        cv2.imwrite(os.path.join(output_dir, f'{name_prefix}_tgt_resized.jpg'), tgt_resized)
    
    # 第一阶段：使用SIFT方法进行粗对齐
    print("\n第二步：使用SIFT方法进行粗对齐...")
    
    def align_with_sift(src_img, tgt_img):
        # 转换为灰度图像
        src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY) if len(src_img.shape) == 3 else src_img
        tgt_gray = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2GRAY) if len(tgt_img.shape) == 3 else tgt_img
        
        # 创建SIFT特征检测器
        sift = cv2.SIFT_create()
        
        # 检测特征点并计算描述符
        kp1, des1 = sift.detectAndCompute(src_gray, None)
        kp2, des2 = sift.detectAndCompute(tgt_gray, None)
        
        print(f"SIFT检测到 - 源图像: {len(kp1)} 个特征点, 目标图像: {len(kp2)} 个特征点")
        
        # 特征匹配
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)
        
        # 应用比率测试筛选好的匹配
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        
        print(f"SIFT找到 {len(good)} 个好的匹配点")
        
        if len(good) < 4:
            print("匹配点不足，无法计算单应性矩阵")
            return None, None, None
        
        # 提取匹配点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        tgt_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        # 使用RANSAC计算单应性矩阵
        H, mask = cv2.findHomography(src_pts, tgt_pts, cv2.USAC_MAGSAC, 5.0)
        
        # 提取内点
        mask = mask.ravel().astype(bool)
        src_pts_inliers = src_pts[mask]
        tgt_pts_inliers = tgt_pts[mask]
        
        print(f"SIFT RANSAC内点: {np.sum(mask)} / {len(good)}")
        
        return H, src_pts_inliers.reshape(-1, 2), tgt_pts_inliers.reshape(-1, 2)
    
    # 执行SIFT对齐
    H_sift, sift_src_pts, sift_tgt_pts = align_with_sift(src_resized, tgt_resized)
    
    if H_sift is None:
        print("SIFT对齐失败，无法继续")
        return None, None
    
    # 可视化SIFT匹配结果
    if output_dir and sift_src_pts is not None and sift_tgt_pts is not None:
        vis_sift = visualize_matches(
            src_resized, tgt_resized,
            sift_src_pts, sift_tgt_pts,
            None,  # 显示所有内点
            os.path.join(output_dir, "stage1", f"{name_prefix}_sift_matches.jpg")
        )
    
    # 应用SIFT变换
    sift_warped_resized = cv2.warpPerspective(src_resized, H_sift, 
                                            (tgt_resized.shape[1], tgt_resized.shape[0]))
    
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, "stage1", f"{name_prefix}_sift_warped.jpg"), sift_warped_resized)
    
    # 第二阶段：使用LoFTR进行细对齐
    print("\n第三步：使用LoFTR进行细对齐...")
    
    def align_with_loftr(src_img, tgt_img):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 转换为灰度图像并准备输入
        src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY) if len(src_img.shape) == 3 else src_img
        tgt_gray = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2GRAY) if len(tgt_img.shape) == 3 else tgt_img
        
        to_tensor = transforms.ToTensor()
        input_dict = {
            "image0": to_tensor(src_gray).unsqueeze(0).to(device),
            "image1": to_tensor(tgt_gray).unsqueeze(0).to(device)
        }
        
        # 加载LoFTR模型
        loftr = LoFTR('outdoor').to(device)
        
        # 运行LoFTR
        with torch.no_grad():
            out = loftr(input_dict)
        
        # 提取关键点和置信度
        keypoints0 = out['keypoints0'].cpu().numpy()
        keypoints1 = out['keypoints1'].cpu().numpy()
        confidence = out['confidence'].cpu().numpy()
        
        print(f"LoFTR找到 {len(keypoints0)} 个匹配点")
        
        # 筛选高置信度的匹配点
        conf_threshold = 0.6
        valid_matches = confidence > conf_threshold
        
        if np.sum(valid_matches) < 4:
            print(f"找到的有效匹配点不足 ({np.sum(valid_matches)} < 4)，无法计算单应性矩阵")
            return None, None, None
        
        # 提取有效的关键点对
        src_pts = keypoints0[valid_matches]
        tgt_pts = keypoints1[valid_matches]
        
        print(f"LoFTR有效匹配点 (置信度 > {conf_threshold}): {np.sum(valid_matches)} / {len(keypoints0)}")
        
        # 使用RANSAC计算单应性矩阵
        H, mask = cv2.findHomography(src_pts, tgt_pts, cv2.USAC_MAGSAC, 3.0)
        
        # 提取内点
        mask = mask.ravel().astype(bool)
        src_pts_inliers = src_pts[mask]
        tgt_pts_inliers = tgt_pts[mask]
        
        print(f"LoFTR RANSAC内点: {np.sum(mask)} / {np.sum(valid_matches)}")
        
        return H, src_pts_inliers, tgt_pts_inliers
    
    # 执行LoFTR对齐
    H_loftr, loftr_src_pts, loftr_tgt_pts = align_with_loftr(sift_warped_resized, tgt_resized)
    
    # 如果LoFTR失败，使用SIFT结果作为最终结果
    if H_loftr is None:
        print("LoFTR对齐失败，使用SIFT结果作为最终结果")
        final_H_resized = H_sift
    else:
        # 可视化LoFTR匹配结果
        if output_dir:
            vis_loftr = visualize_matches(
                sift_warped_resized, tgt_resized,
                loftr_src_pts, loftr_tgt_pts,
                None,  # 显示所有内点
                os.path.join(output_dir, "stage2", f"{name_prefix}_loftr_matches.jpg")
            )
        
        # 组合两个变换矩阵
        final_H_resized = H_loftr @ H_sift
        
        # 应用组合变换
        loftr_warped_resized = cv2.warpPerspective(src_resized, final_H_resized, 
                                                 (tgt_resized.shape[1], tgt_resized.shape[0]))
        
        if output_dir:
            cv2.imwrite(os.path.join(output_dir, "stage2", f"{name_prefix}_loftr_warped.jpg"), loftr_warped_resized)
    
    # 调整变换矩阵以应用于原始尺寸图像
    print("\n第四步：应用最终变换到原始图像...")
    final_H = scale_matrix_dst @ final_H_resized @ scale_matrix_src
    
    # 应用最终变换到原始图像
    final_aligned = cv2.warpPerspective(src_orig, final_H, 
                                      (tgt_orig.shape[1], tgt_orig.shape[0]))
    
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, f"{name_prefix}_final_aligned.jpg"), final_aligned)
        
        
        # 计算SIFT和LoFTR结果的残差
        sift_residual, sift_mean_diff,sift_residual_color = compute_residual(sift_warped_resized, tgt_resized)
        cv2.imwrite(os.path.join(output_dir, "stage1", f"{name_prefix}_sift_residual.jpg"), sift_residual)
        save_image(sift_residual_color, os.path.join(output_dir, "stage1", f"{name_prefix}_sift_residual_color.jpg"))
        print(f"SIFT对齐残差: {sift_mean_diff:.2f}")
        
        if H_loftr is not None:
            loftr_residual, loftr_mean_diff, loftr_residual_color = compute_residual(loftr_warped_resized, tgt_resized)
            cv2.imwrite(os.path.join(output_dir, "stage2", f"{name_prefix}_loftr_residual.jpg"), loftr_residual)
            save_image(loftr_residual_color, os.path.join(output_dir, "stage2", f"{name_prefix}_loftr_residual_color.jpg"))
            print(f"LoFTR对齐残差: {loftr_mean_diff:.2f}")
        
        # 计算最终结果的残差
        final_residual, final_mean_diff,final_residual_color = compute_residual(final_aligned, tgt_orig)
        save_image(final_residual_color, os.path.join(output_dir, f"{name_prefix}_final_residual_color.jpg"))
        cv2.imwrite(os.path.join(output_dir, f"{name_prefix}_final_residual.jpg"), final_residual)
        print(f"最终对齐残差: {final_mean_diff:.2f}")
        
    
    print("\n对齐完成!")
    return final_H, final_aligned


def main():
    # 路径设置
    root_dir = "./TestSet/"
    all_files = os.listdir(root_dir)
    img_names = [file.replace("_200mm.jpg", "") for file in all_files if file.endswith("_200mm.jpg")]
    # 创建基础输出目录
    output_dir = './output_sift_loftr_crop_MAGSAC-denoised'
    os.makedirs(output_dir, exist_ok=True)
    
    for img_name in img_names:
        save_dir = os.path.join(output_dir, img_name)
        os.makedirs(save_dir, exist_ok=True)
        img_200mm_path = os.path.join(root_dir, f'{img_name}_200mm.jpg')
        img_85mm_path = os.path.join(root_dir,  f'{img_name}_85mm.jpg')
        img_35mm_path = os.path.join(root_dir,  f'{img_name}_35mm.jpg')
        img_14mm_path = os.path.join(root_dir,  f'{img_name}_14mm.jpg')
        
        # 图像读取
        img_200mm = cv2.imread(img_200mm_path)
        if img_200mm is None:
            print(f"无法读取图像: {img_200mm_path}")
            return
            
        height, width = img_200mm.shape[:2]
        # 计算旋转矩阵
        # rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 180, 1)
        # img_200mm = cv2.warpAffine(img_200mm, rotation_matrix, (width, height))
        
        img_85mm = cv2.imread(img_85mm_path)
        if img_85mm is None:
            print(f"无法读取图像: {img_85mm_path}")
            return
            
        img_35mm = cv2.imread(img_35mm_path)
        if img_35mm is None:
            print(f"无法读取图像: {img_35mm_path}")
            return
            
        img_14mm = cv2.imread(img_14mm_path)
        if img_14mm is None:
            print(f"无法读取图像: {img_14mm_path}")
            return
    
        # 处理85mm到200mm的对齐
        print("处理85mm到200mm的对齐...")
        H_85_to_200, aligned_85mm = two_stage_alignment(img_85mm, img_200mm, save_dir, "85mm_to_200mm")
        aligned_85mm = cv2.fastNlMeansDenoisingColored(aligned_85mm, None, 10, 10, 7, 21)
        H_85_to_200_refined, aligned_85mm_refined = two_stage_alignment(aligned_85mm, img_200mm, save_dir, "85mm_to_200mm_refined")
        
        # 计算最终变换矩阵并保存最终对齐结果
        final_H_85_to_200 = H_85_to_200_refined @ H_85_to_200
        aligned_85mm_final = cv2.warpPerspective(img_85mm, final_H_85_to_200, (img_200mm.shape[1], img_200mm.shape[0]))
        cv2.imwrite(os.path.join(save_dir, "aligned_85mm_to_200mm.jpg"), aligned_85mm_final)
        
        # 计算并保存残差图
        compute_residual_img_85, mean_diff_85, residual_color_85 = compute_residual(aligned_85mm_final, img_200mm)
        cv2.imwrite(os.path.join(save_dir, "aligned_85mm_to_200mm_residual.jpg"), compute_residual_img_85)
        save_image(residual_color_85, os.path.join(save_dir, "aligned_85mm_to_200mm_residual_color.jpg"))
        
        # 处理35mm到85mm的对齐
        print("处理35mm到85mm的对齐...")
        H_35_to_85, aligned_35mm_to_85 = two_stage_alignment(img_35mm, img_85mm, save_dir, "35mm_to_85mm")
        aligned_35mm_to_85 = cv2.fastNlMeansDenoisingColored(aligned_35mm_to_85, None, 10, 10, 7, 21)
        H_35_to_85_refined, aligned_35mm_to_85_refined = two_stage_alignment(aligned_35mm_to_85, img_85mm, save_dir, "35mm_to_85mm_refined")
        H_35_85_to_200, aligned_35_85_to_200 = two_stage_alignment(aligned_35mm_to_85_refined, img_200mm, save_dir, "35mm_85mm_to_200mm")
        aligned_35_85_to_200 = cv2.fastNlMeansDenoisingColored(aligned_35_85_to_200, None, 10, 10, 7, 21)
        H_35_85_to_200_refined, aligned_35_85_to_200 = two_stage_alignment(aligned_35_85_to_200, img_200mm, save_dir, "35mm_85mm_to_200mm_refined")
        
        # 将35mm直接对齐到200mm
        final_H_35_to_200 = H_35_85_to_200_refined @ H_35_85_to_200 @ H_35_to_85_refined @ H_35_to_85
        aligned_35mm_final = cv2.warpPerspective(img_35mm, final_H_35_to_200, (img_200mm.shape[1], img_200mm.shape[0]))
        cv2.imwrite(os.path.join(save_dir, "aligned_35mm_to_200mm.jpg"), aligned_35mm_final)
        
        # 计算并保存残差图
        compute_residual_img_35, mean_diff_35, residual_color_35 = compute_residual(aligned_35mm_final, img_200mm)
        cv2.imwrite(os.path.join(save_dir, "aligned_35mm_to_200mm_residual.jpg"), compute_residual_img_35)
        save_image(residual_color_35, os.path.join(save_dir, "aligned_35mm_to_200mm_residual_color.jpg"))
        
        # 处理14mm到35mm的对齐
        print("处理14mm到35mm的对齐...")
        H_14_to_35, aligned_14mm_to_35 = two_stage_alignment(img_14mm, img_35mm, save_dir, "14mm_to_35mm")
        aligned_14mm_to_35 = cv2.fastNlMeansDenoisingColored(aligned_14mm_to_35, None, 10, 10, 7, 21)
        H_14_to_35_refined, aligned_14mm_to_35_refined = two_stage_alignment(aligned_14mm_to_35, img_35mm, save_dir, "14mm_to_35mm_refined")
        H_14_35_to_85, aligned_14_35_to_85 = two_stage_alignment(aligned_14mm_to_35_refined, img_85mm, save_dir, "35mm_to_85mm")
        aligned_14_35_to_85 = cv2.fastNlMeansDenoisingColored(aligned_14_35_to_85, None, 10, 10, 7, 21)
        H_14_35_to_85_refined, aligned_14_35_to_85_refined = two_stage_alignment(aligned_14_35_to_85, img_85mm, save_dir, "35mm_to_85mm_refined")
        H_14_35_85_to_200, aligned_14_35_85_to_200 = two_stage_alignment(aligned_14_35_to_85_refined, img_200mm, save_dir, "35mm_85mm_to_200mm")
        aligned_14_35_85_to_200 = cv2.fastNlMeansDenoisingColored(aligned_14_35_85_to_200, None, 10, 10, 7, 21)
        H_14_35_85_to_200_refined, aligned_14_35_85_to_200 = two_stage_alignment(aligned_14_35_85_to_200, img_200mm, save_dir, "35mm_85mm_to_200mm_refined")
        
        
        # 将14mm直接对齐到200mm
        final_H_14_to_200 = H_14_35_85_to_200_refined @ H_14_35_85_to_200 @ H_14_35_to_85_refined @  H_14_35_to_85 @ H_14_to_35_refined @ H_14_to_35
        aligned_14mm_final = cv2.warpPerspective(img_14mm, final_H_14_to_200, (img_200mm.shape[1], img_200mm.shape[0]))
        cv2.imwrite(os.path.join(save_dir, "aligned_14mm_to_200mm.jpg"), aligned_14mm_final)
        
        # 计算并保存残差图
        compute_residual_img_14, mean_diff_14, residual_color_14 = compute_residual(aligned_14mm_final, img_200mm)
        cv2.imwrite(os.path.join(save_dir, "aligned_14mm_to_200mm_residual.jpg"), compute_residual_img_14)
        save_image(residual_color_14, os.path.join(save_dir, "aligned_14mm_to_200mm_residual_color.jpg"))
        
        print("所有对齐处理完成！")
        print(f"85mm到200mm的平均差异: {mean_diff_85}")
        print(f"35mm到200mm的平均差异: {mean_diff_35}")
        print(f"14mm到200mm的平均差异: {mean_diff_14}")
    

if __name__ == "__main__":
    main()
