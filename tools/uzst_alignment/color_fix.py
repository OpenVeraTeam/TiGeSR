import os
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
# 假设这些函数位于 utils/wavelet.py 中 [citation](1)
from utils.wavelet import wavelet_color_fix, adain_color_fix

def process_image(src_path, tgt_path, save_path):
    """
    处理单个图像的函数，用于多进程调用。
    它直接接收源路径、目标路径和保存路径作为参数。
    """
    try:
        src_img = Image.open(src_path)
        tgt_img = Image.open(tgt_path)

        # 色彩校正
        # fixed_img = wavelet_color_fix(src_img, tgt_img)
        fixed_img = adain_color_fix(src_img, tgt_img)

        # 保存处理后的图像
        fixed_img.save(save_path)
        return None  # 成功时返回 None
    except FileNotFoundError:
        return f"Skipped: Target image not found at {tgt_path}"
    except Exception as e:
        # 捕获并记录处理单个文件时可能发生的其他错误
        return f"Failed to process {src_path}: {e}"

def main():
    # --- 配置区域 ---
    tgt_dir = "/data/vjuicefs_ai_camera_llm_ql/intern_data/minxing/results/SR/RealCE/0_3_original_LR_resized"
    root_dir = "/data/vjuicefs_ai_camera_llm_ql/intern_data/minxing/results/SR/RealCE/Ablation_00—worealce-emptymask_8281"
    save_dir = "/data/vjuicefs_ai_camera_llm_ql/intern_data/minxing/results/SR/RealCE/Ablation_00—worealce-emptymask_8281-color-fixed_13mm_adain"
    os.makedirs(save_dir, exist_ok=True)

    tasks = []

    # 1. 收集所有需要处理的任务
    print("Collecting tasks...")
    img_names_with_ext = os.listdir(root_dir)
    for img_name_with_ext in img_names_with_ext:
        # 确保只处理 .png 文件，避免处理其他文件或子目录
        if not img_name_with_ext.endswith(".png"):
            continue

        img_name = os.path.splitext(img_name_with_ext)[0]
        
        src_path = os.path.join(root_dir, f"{img_name}.png")
        tgt_path = os.path.join(tgt_dir, f"{img_name}.png")
        save_path = os.path.join(save_dir, f"{img_name}.png")
        
        tasks.append((src_path, tgt_path, save_path))

    if not tasks:
        print("No tasks to process.")
        return

    # 2. 使用多进程处理任务
    # 自动获取CPU核心数并创建一个合理的进程池
    num_processes = max(1, cpu_count()//2 - 1)
    print(f"Starting processing with {num_processes} processes for {len(tasks)} images...")

    with Pool(processes=num_processes) as pool:
        # 使用 tqdm 显示总体进度
        # pool.starmap 会将 tasks 列表中的每个元组解包作为参数传给 process_image
        results = list(tqdm(pool.starmap(process_image, tasks), total=len(tasks)))

    # 3. （可选）检查并打印处理过程中发生的错误
    errors = [res for res in results if res is not None]
    if errors:
        print("\n--- Issues occurred during processing ---")
        for error in errors:
            print(error)

    print("\nProcessing finished!")

if __name__ == '__main__':
    # 将主程序放在此保护块下是多进程编程的好习惯
    main()
