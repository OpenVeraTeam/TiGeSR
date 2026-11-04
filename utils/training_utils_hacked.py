import argparse
import json
import random
from torch.utils import data as data
from utils.training_utils import FTSRCBDataset,FTSRRealDataset2
from utils.img_util import random_crop_tensors, random_crop_multiple_images
from torchvision import transforms
import os
from PIL import Image
import cv2
import torchvision.transforms.functional as F
from easydict import EasyDict as edict
from models.ocr_recog.RecModel import RecModel
import torch
import numpy as np


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, monitor='eval/lpips', mode='min'):
        """
        Args:
            patience (int): 在多少个评估周期后没有改善就停止训练
            min_delta (float): 被认为是改善的最小变化
            monitor (str): 要监控的指标名称
            mode (str): 'min' 表示指标越小越好，'max' 表示指标越大越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif self._is_improvement(current_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, current_score):
        if self.mode == 'min':
            return current_score < self.best_score - self.min_delta
        else:  # mode == 'max'
            return current_score > self.best_score + self.min_delta


# CTR数据集类定义与处理
class RealCEDataset(data.Dataset):
    def __init__(self, dataset_folder: str,use_hisam=True):
        super().__init__()
        self.input_folder = dataset_folder
        self.img_names = os.listdir(os.path.join(self.input_folder,"13mm_aligned_1024"))
        # self.img_names = os.listdir(os.path.join(self.input_folder,"52mm_deg"))
        self.use_hisam = use_hisam
        # print("Whether to use hisam:",self.use_hisam)
        self.default_dict = None
        self.store_flag = False

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]
        try:
            lr_path = os.path.join(self.input_folder, "13mm_aligned_1024", name)
            # lr_path = os.path.join(self.input_folder, "52mm_deg", name)
            gt_path = os.path.join(self.input_folder, "52mm_aligned_1024", name)
            mask_path = os.path.join(
                    self.input_folder,
                    "empty_mask",
                    name.split(".")[0] + ".png",
            )
            # mask_path = os.path.join(
            #         self.input_folder,
            #         "Ours_224000_135200_1Out",
            #         name.split(".")[0] + "_mask.png",
            # )
            # mask_path = os.path.join(
            #         self.input_folder,
            #         "mask_std",
            #         name.split(".")[0] + ".png",
            # )
            # mask_path = os.path.join(
            #         self.input_folder,
            #         "mask-13mm-bts-1024",
            #         name.split(".")[0] + ".png"
            # )
            # gt_path = os.path.join(self.input_folder, "52mm", name)
            # if self.use_hisam:
            #     mask_path = os.path.join(
            #         self.input_folder,
            #         "mask_52mm_patchedx4",
            #         name.split(".")[0] + ".png",
            #     )
                
            # else:
            #     mask_path = os.path.join(self.input_folder, "Ours_372000_118131", name.split(".")[0] + "_mask.png")
            # mask_path = os.path.join(self.input_folder, "Ours_464000_118131_dilate21", name.split(".")[0] + "_mask.png")
            # mask_path = os.path.join(self.input_folder, "hisam", name.split(".")[0] + ".png")

            input_img = Image.open(lr_path).convert("RGB")
            gt_img = Image.open(gt_path).convert("RGB")
            mask_img = Image.open(mask_path).convert("RGB")
            w,h = input_img.size
            ra = 512 / max(w,h)
            # input_img = input_img.resize((round(w*ra),round(h*ra)))
            # gt_img = gt_img.resize((round(w*ra),round(h*ra)))
            # mask_img = mask_img.resize((round(w*ra),round(h*ra)))

            input_img = F.to_tensor(input_img)
            gt_img = F.to_tensor(gt_img)
            mask_img = F.to_tensor(mask_img)
            # mask_img = torch.zeros_like(gt_img)
            # bbox = self.bbox_data[name + ".png"]
            
            res = {
                "name": os.path.splitext(os.path.basename(name))[0],
                "input": input_img,
                "gt": gt_img,
                "mask": mask_img,
                # "bbox": bbox
            }

            if not self.store_flag:
                self.default_dict = res
                self.store_flag = True
                # self.default_dict['name'] = "ERROR"
            return res
        except Exception as e:
            print(f"Error loading image {name}: {e}")
            return self.default_dict

class UltraLensDataset(data.Dataset):
    def __init__(self, dataset_folder: str, Splited_txt:str):
        super().__init__()
        self.input_folder = dataset_folder
        self.img_names = self._load_file_names(Splited_txt)
        self.img_names = sorted(self.img_names)
        self.lr_folder = os.path.basename(Splited_txt).split('_')[0]
        self.default_dict = None
        self.store_flag = False

    def __len__(self):
        return len(self.img_names)
    def _load_file_names(self, txt_path: str) -> list:
        """从txt文件中读取文件名列表"""
        file_names = []
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # 去除每行的换行符和空白字符
                    filename = line.strip()
                    if filename:  # 跳过空行
                        file_names.append(filename)
        except FileNotFoundError:
            try: 
                txt_path = os.path.join(self.input_folder,txt_path)
                with open(txt_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        # 去除每行的换行符和空白字符
                        filename = line.strip()
                        if filename:  # 跳过空行
                            file_names.append(filename)
            except FileNotFoundError:
                print(f"错误：找不到文件 {txt_path}")
                return []
        except Exception as e:
            print(f"读取文件时出错：{e}")
            return []
        
        return file_names
    
    def __getitem__(self, idx):
        name = self.img_names[idx]
        try:
            lr_path = os.path.join(self.input_folder,self.lr_folder, name+ ".png")
            gt_path = os.path.join(self.input_folder, "200mm", name+ ".png")
            mask_path = os.path.join(
                    self.input_folder,
                    "mask",
                    name.split(".")[0] + ".png",
                )

            input_img = Image.open(lr_path).convert("RGB")
            gt_img = Image.open(gt_path).convert("RGB")
            mask_img = Image.open(mask_path).convert("RGB")
            w,h = input_img.size
            ra = 1024 / max(w,h)
            input_img = input_img.resize((round(w*ra)//8*8,round(h*ra)//8*8))
            gt_img = gt_img.resize((round(w*ra)//8*8,round(h*ra)//8*8))
            mask_img = mask_img.resize((round(w*ra)//8*8,round(h*ra)//8*8))

            input_img = F.to_tensor(input_img)
            gt_img = F.to_tensor(gt_img)
            mask_img = F.to_tensor(mask_img)
            
            res = {
                "name": os.path.splitext(os.path.basename(name))[0],
                "input": input_img,
                "gt": gt_img,
                "mask": mask_img,
                # "bbox": bbox
            }

            if not self.store_flag:
                self.default_dict = res
                self.store_flag = True
                # self.default_dict['name'] = "ERROR"
            return res
        except Exception as e:
            print(f"Error loading image {name}: {e}")
            return self.default_dict


class RealCECropDataset(RealCEDataset):
    def __init__(self, dataset_folder: str):
        super().__init__(dataset_folder)
        self.min_scale=2
        self.max_scale=8

    def __getitem__(self, idx):
        name = self.img_names[idx]
        try:
            lr_path = os.path.join(self.input_folder, "13mm_aligned_1024", name)
            gt_path = os.path.join(self.input_folder, "52mm_aligned_1024", name)
            mask_path = os.path.join(
                self.input_folder,
                "mask_52mm_patchedx4",
                name.split(".")[0] + ".png",
            )
            input_img = Image.open(lr_path).convert("RGB")
            gt_img = Image.open(gt_path).convert("RGB")
            mask_img = Image.open(mask_path).convert("RGB")
            width, height = input_img.size
            
            scale_factor = random.uniform(self.min_scale, self.max_scale)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            input_img = input_img.resize((new_width, new_height), Image.LANCZOS)
            gt_img = gt_img.resize((new_width, new_height), Image.LANCZOS)
            mask_img = mask_img.resize((new_width, new_height), Image.LANCZOS)
            [input_img, gt_img, mask_img] = (
                random_crop_multiple_images(
                    [input_img, gt_img, mask_img],
                    crop_size=(height // 8 * 8, width // 8 * 8),
                )
            )

            input_img = F.to_tensor(input_img)
            gt_img = F.to_tensor(gt_img)
            mask_img = F.to_tensor(mask_img)
            # bbox = self.bbox_data[name + ".png"]

            res = {
                "name": name.split(".")[0],
                "input": input_img,
                "gt": gt_img,
                "mask": mask_img,
                # "bbox": bbox
            }

            if not self.store_flag:
                self.default_dict = res
                self.store_flag = True
                # self.default_dict['name'] = "ERROR"
            return res
        except Exception as e:
            print(f"Error loading image {name}: {e}")
            return self.default_dict


class UltraLensCropDataset(UltraLensDataset):
    def __init__(self,  dataset_folder: str, Splited_txt:str):
        super().__init__(dataset_folder,Splited_txt)
        self.min_scale=2
        self.max_scale=8

    def __getitem__(self, idx):
        name = self.img_names[idx]
        try:
            lr_path = os.path.join(self.input_folder,self.lr_folder, name+ ".png")
            gt_path = os.path.join(self.input_folder, "200mm", name+ ".png")
            mask_path = os.path.join(
                    self.input_folder,
                    "mask",
                    name.split(".")[0] + ".png",
                )

            input_img = Image.open(lr_path).convert("RGB")
            gt_img = Image.open(gt_path).convert("RGB")
            mask_img = Image.open(mask_path).convert("RGB")
            w,h = input_img.size
            ra = 1024 / max(w,h)
            height = round(h*ra)
            width  = round(w*ra)
            scale_factor = random.uniform(self.min_scale, self.max_scale)
            input_img = input_img.resize((round(w*ra*scale_factor)//8*8,round(h*ra*scale_factor)//8*8))
            gt_img = gt_img.resize((round(w*ra*scale_factor)//8*8,round(h*ra*scale_factor)//8*8))
            mask_img = mask_img.resize((round(w*ra*scale_factor)//8*8,round(h*ra*scale_factor)//8*8))
            [input_img, gt_img, mask_img] = (
                random_crop_multiple_images(
                    [input_img, gt_img, mask_img],
                    crop_size=(height // 8 * 8, width // 8 * 8),
                )
            )

            input_img = F.to_tensor(input_img)
            gt_img = F.to_tensor(gt_img)
            mask_img = F.to_tensor(mask_img)
            
            res = {
                "name": os.path.splitext(os.path.basename(name))[0],
                "input": input_img,
                "gt": gt_img,
                "mask": mask_img,
                # "bbox": bbox
            }

            if not self.store_flag:
                self.default_dict = res
                self.store_flag = True
                # self.default_dict['name'] = "ERROR"
            return res
        except Exception as e:
            print(f"Error loading image {name}: {e}")
            return self.default_dict

# LSDIR数据集处理类
class LSDIRsyn(FTSRCBDataset):
    def __init__(self, dataset_folder, min_no=0, max_no=500,std=False):
        super().__init__(dataset_folder, min_no, max_no)
        self.std = std

    def __getitem__(self, idx):
        name = '{}'.format(str(idx + self.min_no + 1).zfill(7))
        try:
            lr_path = os.path.join(self.input_folder, "sr_bicubic", name + ".png")
            gt_path = os.path.join(self.input_folder, "gt", name + ".png")
            if self.std:
                mask_path = os.path.join(self.input_folder, "mask_std", name + ".png")
            else:
                mask_path = os.path.join(self.input_folder, "mask", name + ".png")

            input_img = Image.open(lr_path).convert("RGB")
            gt_img = Image.open(gt_path).convert("RGB")
            mask_img = Image.open(mask_path).convert("RGB")

            input_img = F.to_tensor(input_img)
            gt_img = F.to_tensor(gt_img)
            mask_img = F.to_tensor(mask_img)
            # bbox = self.bbox_data[name + ".png"]
            res = {
                "name": name,
                "input": input_img,
                "gt": gt_img,
                "mask": mask_img,
                # "bbox": bbox
            }

            if not self.store_flag:
                self.default_dict = res
                self.store_flag = True
                # self.default_dict['name'] = "ERROR"
            return res
        except Exception as e:
            print(f"Error loading image {name}: {e}")
            return self.default_dict

# LSDIR数据集测试类
class LSDIRsynTest(FTSRRealDataset2):
    def __init__(self, dataset_folder:str):
        super().__init__(dataset_folder=dataset_folder)
        self.default_dict = None
        self.store_flag = False
    
    def __getitem__(self, idx):
        name = self.img_names[idx]
        try:
            lr_path = os.path.join(self.input_folder, "sr_bicubic", name)
            gt_path = os.path.join(self.input_folder, "gt", name)
            mask_path = os.path.join(self.input_folder, "mask", name.split('.')[0] + ".png")

            input_img = Image.open(lr_path).convert("RGB")
            gt_img = Image.open(gt_path).convert("RGB")
            mask_img = Image.open(mask_path).convert("RGB")

            input_img = F.to_tensor(input_img)
            gt_img = F.to_tensor(gt_img)
            mask_img = F.to_tensor(mask_img)
            # bbox = self.bbox_data[name + ".png"]
            res = {
                "name": name,
                "input": input_img,
                "gt": gt_img,
                "mask": mask_img,
                # "bbox": bbox
            }

            if not self.store_flag:
                self.default_dict = res
                self.store_flag = True
                # self.default_dict['name'] = "ERROR"
            return res
        except Exception as e:
            print(f"Error loading image {name}: {e}")
            return self.default_dict

class LSDIRsyn_canny(LSDIRsyn):
    def __getitem__(self, idx):
        name = '{}'.format(str(idx + self.min_no + 1).zfill(7))
        try:
            lr_path = os.path.join(self.input_folder, "sr_bicubic", name + ".png")
            gt_path = os.path.join(self.input_folder, "gt", name + ".png")
            mask_path = os.path.join(self.input_folder, "canny", name + ".png")

            input_img = Image.open(lr_path).convert("RGB")
            gt_img = Image.open(gt_path).convert("RGB")
            mask_img = Image.open(mask_path).convert("RGB")

            input_img = F.to_tensor(input_img)
            gt_img = F.to_tensor(gt_img)
            mask_img = F.to_tensor(mask_img)

            # bbox = self.bbox_data[name + ".png"]
            res = {
                "name": name,
                "input": input_img,
                "gt": gt_img,
                "mask": mask_img,
                # "bbox": bbox
            }

            if not self.store_flag:
                self.default_dict = res
                self.store_flag = True
                # self.default_dict['name'] = "ERROR"
            return res
        except Exception as e:
            print(f"Error loading image {name}: {e}")
            return self.default_dict

class LSDIRsyn_canny_test(LSDIRsynTest):
    def __getitem__(self, idx):
        name = self.img_names[idx]
        try:
            lr_path = os.path.join(self.input_folder, "sr_bicubic", name)
            gt_path = os.path.join(self.input_folder, "gt", name)
            mask_path = os.path.join(self.input_folder, "canny", name.split('.')[0] + ".png")

            input_img = Image.open(lr_path).convert("RGB")
            gt_img = Image.open(gt_path).convert("RGB")
            mask_img = Image.open(mask_path).convert("RGB")

            input_img = F.to_tensor(input_img)
            gt_img = F.to_tensor(gt_img)
            mask_img = F.to_tensor(mask_img)
            # bbox = self.bbox_data[name + ".png"]
            res = {
                "name": name,
                "input": input_img,
                "gt": gt_img,
                "mask": mask_img,
                # "bbox": bbox
            }

            if not self.store_flag:
                self.default_dict = res
                self.store_flag = True
                # self.default_dict['name'] = "ERROR"
            return res
        except Exception as e:
            print(f"Error loading image {name}: {e}")
            return self.default_dict

def parse_args_paired_training(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()
    # args for continue training
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--sd3_dir", default="weights/stable-diffusion-3.5-medium")
    parser.add_argument("--prompt_cache_path", default=None)

    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_lpips", default=5, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_ocr", default=1000, type=float)
    parser.add_argument("--lambda_ctc", default=1.0, type=float)
    parser.add_argument("--lambda_edge", default=1.0, type=float)
    parser.add_argument("--lambda_bce", default=1.0, type=float)
    parser.add_argument("--lambda_clipsim", default=5.0, type=float)
    parser.add_argument("--timestep", default=500, type=float)
    parser.add_argument("--ctrl_layers", default=12, type=int)
    parser.add_argument("--cond_scale", default=1.0, type=float)

    # dataset options
    parser.add_argument("--train_folders", type=str)
    parser.add_argument("--test_folder", type=str)

    parser.add_argument("--train_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--test_image_prep", default="resized_crop_512", type=str)

    # validation eval args
    parser.add_argument("--eval_freq", default=20, type=int)
    parser.add_argument("--track_val_fid", default=False, action="store_true")
    parser.add_argument("--num_samples_eval", type=int, default=100, help="Number of samples to use for all evaluation")
    parser.add_argument("--num_samples_save", type=int, default=20, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--tracker_project_name", type=str, default="train_pix2pix_turbo",
                        help="The name of the wandb project to log to.")

    # details about the model architecture
    parser.add_argument("--lora_rank_unet", default=8, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None, )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512, )
    parser.add_argument("--train_batch_size", type=int, default=1,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=50)
    parser.add_argument("--max_train_steps", type=int, default=10_000, )
    parser.add_argument("--checkpointing_steps", type=int, default=500, )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--gradient_checkpointing", action="store_true", )
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help=(
                            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                            ' "constant", "constant_with_warmup"]'
                        ),
                        )
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
                        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
                        )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0, )
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        help=(
                            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
                            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
                        ),
                        )
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true",
                        help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true", )
    parser.add_argument("--has_vae", action="store_true", )
    parser.add_argument("--use_std", action="store_true", )
    # 在parse_args_paired_training函数中添加以下参数
    parser.add_argument("--early_stopping", action="store_true", help="启用early stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Early stopping的耐心值")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.001, help="Early stopping的最小改善阈值")
    parser.add_argument("--early_stopping_monitor", type=str, default="eval/lpips", choices=["eval/lpips", "eval/psnr", "eval/ssim"], help="Early stopping监控的指标")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # args.train_folders = args.train_folders.split(',')
    return args

def parse_args_test(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()
    # args for continue training
    parser.add_argument("--ctrlnet_ckpt_path", default=None)
    parser.add_argument("--idm_ckpt_path", default=None)
    parser.add_argument("--enc_path", default="weights/DiffTSR/Bert_only.ckpt")
    parser.add_argument("--anno_path", default=None, type=str,
                        help="Path to the annotation file for the input images. If None, no annotations will be used.")
    parser.add_argument("--enable_tiling", action="store_true")
    # args for the loss function
    parser.add_argument("--timestep", default=500, type=float)
    parser.add_argument("--ctrl_layers", default=12, type=int)
    parser.add_argument("--cond_scale", default=1.0, type=float)
    parser.add_argument(
        "--difftsr_config", default="./configs/DiffTSR_mask_config.yaml", type=str
    )

    # dataset options
    parser.add_argument("--test_folder", type=str)

    parser.add_argument("--train_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--test_image_prep", default="resized_crop_512", type=str)

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512, )
    parser.add_argument("--gradient_checkpointing", action="store_true", )

    parser.add_argument("--dataloader_num_workers", type=int, default=0, )
    parser.add_argument("--set_grads_to_none", action="store_true", )
    parser.add_argument("--has_vae", action="store_true", )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # args.train_folders = args.train_folders.split(',')
    return args

def parse_args_lora_training(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()
    # args for continue training
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--pretrained_model_name_or_path", default="./weights/stable-diffusion-3.5-medium")

    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_lpips", default=5, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_edge", default=1.0, type=float)
    parser.add_argument("--lambda_clipsim", default=5.0, type=float)
    parser.add_argument("--timestep", default=500, type=float)
    # Lora Params
    parser.add_argument("--rank", default=64, type=int)
    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            "The transformer block layers to apply LoRA training on. Please specify the layers in a comma seperated string."
            "For examples refer to https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_SD3.md"
        ),
    )
    parser.add_argument(
        "--lora_blocks",
        type=str,
        default=None,
        help=(
            "The transformer blocks to apply LoRA training on. Please specify the block numbers in a comma seperated manner."
            'E.g. - "--lora_blocks 12,30" will result in lora training of transformer blocks 12 and 30. For more examples refer to https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_SD3.md'
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )
    # ControlNet Params
    parser.add_argument("--ctrl_layers", default=12, type=int)
    parser.add_argument("--cond_scale", default=1.0, type=float)
    parser.add_argument("--num_extra_conditioning_channels", default=0, type=int)
    parser.add_argument("--controlnet_learning_rate", default=5e-6, type=int)


    # dataset options
    parser.add_argument("--train_folders", type=str)
    parser.add_argument("--test_folder", type=str)

    parser.add_argument("--train_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--test_image_prep", default="resized_crop_512", type=str)

    # validation eval args
    parser.add_argument("--eval_freq", default=20, type=int)
    parser.add_argument("--track_val_fid", default=False, action="store_true")
    parser.add_argument("--num_samples_eval", type=int, default=100, help="Number of samples to use for all evaluation")
    parser.add_argument("--num_samples_save", type=int, default=20, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--tracker_project_name", type=str, default="train_pix2pix_turbo",
                        help="The name of the wandb project to log to.")


    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None, )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512, )
    parser.add_argument("--train_batch_size", type=int, default=1,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=50)
    parser.add_argument("--max_train_steps", type=int, default=200000, )
    parser.add_argument("--checkpointing_steps", type=int, default=500, )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--gradient_checkpointing", action="store_true", )
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help=(
                            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                            ' "constant", "constant_with_warmup"]'
                        ),
                        )
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
                        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
                        )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0, )
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        help=(
                            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
                            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
                        ),
                        )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true",
    #                     help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true", )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # args.train_folders = args.train_folders.split(',')
    return args


def create_predictor(model_dir=None, model_lang='ch', is_onnx=False):
    model_file_path = model_dir
    if model_file_path is not None and not os.path.exists(model_file_path):
        raise ValueError("not find model file path {}".format(model_file_path))

    if is_onnx:
        import onnxruntime as ort
        sess = ort.InferenceSession(model_file_path, providers=['CPUExecutionProvider'])  # 'TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'
        return sess
    else:
        if model_lang == 'ch':
            n_class = 6625
        elif model_lang == 'en':
            n_class = 97
        elif model_lang == 'v5':
            n_class = 18385
        else:
            raise ValueError(f"Unsupported OCR recog model_lang: {model_lang}")
        rec_config = edict(
            in_channels=3,
            backbone=edict(type='MobileNetV1Enhance', scale=0.5, last_conv_stride=[1, 2], last_pool_type='avg'),
            neck=edict(type='SequenceEncoder', encoder_type="svtr", dims=64, depth=2, hidden_dims=120, use_guide=True),
            head=edict(type='CTCHead', fc_decay=0.00001, out_channels=n_class, return_feats=True)
        )

        rec_model = RecModel(rec_config)
        if model_file_path is not None:
            rec_model.load_state_dict(torch.load(model_file_path, map_location="cpu"))
            rec_model.eval()
        return rec_model.eval()
