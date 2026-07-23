import cv2
import numpy as np
import os
import torch
from torchvision.utils import save_image
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from torchvision.transforms import ToTensor, ToPILImage
from utils.wavelet import adaptive_instance_normalization, wavelet_color_fix,wavelet_color_fix_cv2
import pymagsac

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
    snn_ratios= [m.distance / n.distance for m, n in matches if m.distance < threshold * n.distance]

    if len(good) < 4:
        print("匹配点不足，跳过对齐")
        return None
    sorted_indices = np.argsort(snn_ratios)
    tentatives = list(np.array(good)[sorted_indices])
    correspondences = np.float32([ (kp1[m.queryIdx].pt + kp2[m.trainIdx].pt) for m in tentatives ]).reshape(-1,4)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = pymagsac.findHomography(
        np.ascontiguousarray(correspondences), 
        src_img.shape[0], src_img.shape[1], tgt_img.shape[0], tgt_img.shape[1],
        probabilities = [],
        sampler = 2,
        use_magsac_plus_plus = True,
        sigma_th = 2.0)
    # H, _ = cv2.findHomography(pts1, pts2, cv2.USAC_MAGSAC, 5.0)
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
        H_refined = align_image(current_img, target_img, threshold=Fine_thresh)
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
        imwrite_unicode(os.path.join(cascade_alignment_dir, f"{img_name}_{name_prefix}.png"), final_aligned_img)
    
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
    # ==================== 新增代码段开始 ====================
    # 1. 定义包含待处理文件列表的txt文件路径
    unaligned_txt_path = 'output/UltraLens/sift_ripe_MAGSAC_corrected/supp.txt'
    
    # 2. 读取txt文件，并将文件名存储到一个set中以便快速查找
    #    使用try-except处理文件不存在的情况
    try:
        with open(unaligned_txt_path, 'r') as f:
            # 使用strip()去除每行末尾的换行符
            # 使用set来存储，可以自动去重且查找速度快
            files_to_process = {line.replace("_35mm","").replace("_85mm","").replace("_14mm","").strip() for line in f}
        print(f"成功加载 {len(files_to_process)} 个待处理文件名从: {unaligned_txt_path}")
    except FileNotFoundError:
        print(f"错误: 待处理文件列表 '{unaligned_txt_path}' 未找到。程序将退出。")
        return # 如果文件不存在，直接退出程序
    # ==================== 新增代码段结束 ====================
    # 创建基础输出目录
    save_dir = './output/UltraLens_supp/1siftinChain_1siftinFine_MAGSAC++'
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建子目录
    original_dir = os.path.join(save_dir, "original")
    cascade_alignment_dir = os.path.join(save_dir, "cascade_alignment")
    residual_dir = os.path.join(save_dir, "residual")
    
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(cascade_alignment_dir, exist_ok=True)
    os.makedirs(residual_dir, exist_ok=True)
    
    # 图像读取
    # root_dir = 'TestSet'
    root_dir = '/data/vjuicefs_ai_camera_llm/intern_data/minxing/datasets/UltraLens/original'
    all_files = os.listdir(root_dir)
    img_names = [file.replace("_200mm_cr.png", "") for file in all_files if file.endswith("_200mm_cr.png")]
    img_names = sorted(set(img_names))  # 去重并排序
    
    for img_name in img_names:
        # ==================== 修改部分开始 ====================
        # 3. 检查当前文件名是否在待处理列表中
        if img_name not in files_to_process:
            # 如果不在，打印信息并跳过此次循环
            # print(f"Skipping '{img_name}' as it is not in {unaligned_txt_path}") # 可以取消注释这行来获得更详细的日志
            continue
        # ==================== 修改部分结束 ====================
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
