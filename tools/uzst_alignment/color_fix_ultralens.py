import os
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
# 假设这些函数位于 utils/wavelet.py 中
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
        fixed_img = wavelet_color_fix(src_img, tgt_img)
        
        # 保存处理后的图像
        fixed_img.save(save_path)
        return None
    except Exception as e:
        return f"Failed to process {src_path}: {e}"

def main():
    Lenses = [
        "14mm", 
        "35mm", 
        # "85mm"
    ]
    method = "1Ours_45852_186000_no_sigmoid"
    
    tasks = []
    
    # 1. 收集所有需要处理的任务
    print("Collecting tasks...")
    for Lens in Lenses:
        tgt_dir = f"/data/vjuicefs_ai_camera_llm/intern_data/minxing/results/SR/UltraLens_filtered/{Lens}/0_1_LR"
        root_dir = f"/data/vjuicefs_ai_camera_llm/intern_data/minxing/results/SR/UltraLens_filtered/{Lens}/{method}"
        save_dir = f"/data/vjuicefs_ai_camera_llm/intern_data/minxing/results/SR/UltraLens_filtered/{Lens}/{method}-color-fixed_13mm_wavlet"
        os.makedirs(save_dir, exist_ok=True)
        
        img_names = os.listdir(root_dir)
        for img_name_with_ext in img_names:
            # 忽略非图像文件或特定目录
            if not img_name_with_ext.endswith(".png"):
                continue
            
            img_name = os.path.splitext(img_name_with_ext)[0]
            
            src_path = os.path.join(root_dir, f"{img_name}.png")
            tgt_path = os.path.join(tgt_dir, f"{img_name}.png")
            save_path = os.path.join(save_dir, f"{img_name}.png")
            
            # 检查目标文件是否存在，避免重复处理
            if not os.path.exists(tgt_path):
                print(f"Warning: Target image not found, skipping: {tgt_path}")
                continue

            tasks.append((src_path, tgt_path, save_path))

    if not tasks:
        print("No tasks to process.")
        return

    # 2. 使用多进程处理任务
    # 使用 cpu_count() - 1 个进程，留一个核心给系统，避免卡顿
    num_processes = max(1, cpu_count() // 4 - 1)
    print(f"Starting processing with {num_processes} processes for {len(tasks)} images...")
    
    with Pool(processes=num_processes) as pool:
        # 使用 tqdm 显示总体进度
        # pool.starmap 会将 tasks 列表中的每个元组解包作为参数传给 process_image
        results = list(tqdm(pool.starmap(process_image, tasks), total=len(tasks)))
    
    # 3. （可选）检查处理结果
    errors = [res for res in results if res is not None]
    if errors:
        print("\n--- Errors occurred during processing ---")
        for error in errors:
            print(error)
    
    print("\nProcessing finished!")

if __name__ == '__main__':
    # 在 Windows 或 macOS 上，建议将多进程启动代码放在 if __name__ == '__main__': 块中
    main()
