import gc
import json
import os

import torch
import torch.utils.checkpoint
import torch.utils.data
from tqdm.auto import tqdm
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, "..")))

from models.FullPipeline import TextSR_multiscale
from utils.training_utils_hacked import parse_args_test
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms.functional as tF
import torch.nn.functional as F

import torch.multiprocessing as mp

# 每个进程的处理函数
def process_images(rank, world_size, args, img_names):
    # 设置设备
    torch.cuda.set_device(rank)
    
    # 初始化模型
    net_sr = TextSR_multiscale(
        args.difftsr_config,
        target_timestep=args.timestep,
        ckpt_dir="./weights/stable-diffusion-3.5-medium",
        gradient_checkpointing=args.gradient_checkpointing,
        num_ctrl_layers=args.ctrl_layers,
    )
    net_sr.load_model(args.idm_ckpt_path, args.ctrlnet_ckpt_path, args.enc_path)
    net_sr.set_eval()
    net_sr.to_device(rank)
    
    # 为当前进程分配图像
    local_img_names = img_names[rank::world_size]
    
    for img_name in local_img_names:
        output_path = os.path.join(args.output_dir, img_name.split('.')[0] + ".png")
        if os.path.exists(output_path):
            print(f"文件 {output_path} 已存在，跳过处理")
            continue
        # if img_name!="IMG_20250724_173355_1_14mm.png":
        #     print(f"文件 {output_path} 已存在，跳过处理")
        #     continue
            
        lq_path = os.path.join(args.test_folder, img_name)
        input_img = Image.open(lq_path).convert("RGB")
        W, H = input_img.size
        ratio = 1024/max(W,H)
        # ratio = 1
        input_img = input_img.resize((round(W * ratio) // 8 * 8, round(H * ratio) // 8 * 8))
        # input_img = input_img.resize((W // 8 * 8, H // 8 * 8))  # 确保尺寸是8的倍数
        input_img = tF.to_tensor(input_img).unsqueeze(0).to(f"cuda:{rank}")
        
        with torch.no_grad():
            print(f"GPU {rank} 处理: {img_name}")
            if args.anno_path is not None:
                anno_file = os.path.join(args.anno_path, img_name.split('.')[0] + '_res.json')
                with open(anno_file, 'r') as f:
                    anno = json.load(f)
                sr_img, mask, sub_sr = net_sr.inference_wanno(input_img, anno,tiling=args.enable_tiling)
            else:
                sr_img, mask, sub_sr = net_sr.inference(input_img,anno=None,tiling=args.enable_tiling)
                
            # 保存结果
            os.makedirs(args.output_dir, exist_ok=True)
            # save_image(input_img.cpu(), os.path.join(args.output_dir, img_name.split('.')[0] + "_lr.png"))
            save_image(sr_img.cpu(), os.path.join(args.output_dir, img_name.split('.')[0] + ".png"))
            save_image(mask.cpu(), os.path.join(args.output_dir, img_name.split('.')[0] + "_mask.png"))
            save_image(sub_sr.cpu(), os.path.join(args.output_dir, img_name.split('.')[0] + "_subsr.png"))
            
        # 清理内存
        gc.collect()
        torch.cuda.empty_cache()

# 主函数
def main(args):
    # 获取可用GPU数量
    num_gpus = torch.cuda.device_count()
    # num_gpus = 2
    if num_gpus < 1:
        raise RuntimeError("没有可用的GPU")
    
    print(f"使用 {num_gpus} 个GPU进行并行推理")
    
    # 获取所有图像名称
    img_names = [f for f in os.listdir(args.test_folder) 
                if f.endswith(('.jpg', '.png', '.JPG'))]
    img_names.sort()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 启动多进程
    mp.spawn(
        process_images,
        args=(num_gpus, args, img_names),
        nprocs=num_gpus,
        join=True
    )

if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    args = parse_args_test()
    # 添加分布式参数
    args.distributed = False  # 这里我们使用多进程而非DDP
    
    main(args)
