import math
from re import sub
import cv2
import numpy as np
import torch.nn as nn

import torch
from tqdm import tqdm
from models.DiffTSR.IDM.utils.util import instantiate_from_config
from models.pure_ctrlnet import SRCtrlnetPipeline
from models.Skip_con import SRCtrlnetSkipConnectionPipeline
from omegaconf import OmegaConf
from models.recognizer import adjust_image, restore_image,restore_image1,restore_image_cv2,adjust_image_cv2
import argparse
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from paddleocr import PaddleOCR
from utils.img_util import (
    remove_padding,
    resize_and_pad,
    resize_and_cat,
    split_image_with_overlap,
    merge_blocks_with_overlap,
    extract_overlapping_patches,
    combine_sr_patches_weighted,
    combine_mask_patches,
    split_long_image_tensor,
    split_text_annotation,
    merge_image_blocks
)
from kornia.morphology import dilation, erosion


def paste_sub_image_with_pil(hq: torch.Tensor, sub_sr: torch.Tensor, box: np.ndarray) -> torch.Tensor:
    """
    使用 PIL.Image.paste 将 sub_sr 的内容粘贴到 hq 的指定 box 区域。

    参数:
    - hq (torch.Tensor): 基础高清图像 (C, H, W)，数值范围 [0, 1]。
    - sub_sr (torch.Tensor): 包含待粘贴内容的小块超分图像 (C, H, W)，
                             通常在 box 区域外是黑色的。数值范围 [0, 1]。
    - box (np.ndarray): 四边形顶点坐标，形状为 (4, 2)，如 [[x1, y1], ...]。

    返回:
    - torch.Tensor: 融合后的图像 (C, H, W)。
    """
    # 1. 将 PyTorch Tensor 转换为 PIL Image
    # TF.to_pil_image 会自动处理 (C, H, W) -> (H, W, C) 的转换
    hq_pil = TF.to_pil_image(hq)
    
    # 2. 计算外接矩形 (Bounding Box) 和粘贴位置
    # box 的坐标格式为 (x, y)，我们需要 x_min, y_min, x_max, y_max
    x_min, y_min = np.min(box, axis=0).astype(int)
    x_max, y_max = np.max(box, axis=0).astype(int)
    
    # 粘贴位置是外接矩形的左上角
    paste_position = (x_min, y_min)
    
    # 3. 从 sub_sr 中裁剪出有效的图像块
    # 注意：Tensor的裁剪顺序是 [C, H, W]，所以是 y_min:y_max, x_min:x_max
    sub_sr_patch = sub_sr[:, y_min:y_max, x_min:x_max]
    sub_sr_patch_pil = TF.to_pil_image(sub_sr_patch)
    
    # 4. 执行粘贴操作
    # paste(im, box, mask=None)
    # 这里的 mask 参数可以用来处理非矩形粘贴，如果 sub_sr_patch_pil 有 alpha 通道，
    # paste 会自动使用它作为蒙版。
    hq_pil.paste(sub_sr_patch_pil, paste_position)
    
    # 5. 将结果转回 PyTorch Tensor
    hq_fused_tensor = TF.to_tensor(hq_pil).to(hq.device)
    
    return hq_fused_tensor

class DiffTSRRestore(nn.Module):

    def __init__(
        self,
        DiffTSR_yaml_config="./models/DiffTSR/DiffTSR_config.yaml",
    ):
        super().__init__()
        self.execution_device = torch.device("cuda")
        raise RuntimeError("DiffTSRRestore is not part of the released TIGER pipeline. Use TextSR_multiscale instead.")
        DiffTSR_config = OmegaConf.load(DiffTSR_yaml_config)
        self.net_sub_gen = instantiate_from_config(DiffTSR_config.model)

    def load_model(
        self,
        idm_ckpt,
    ):
        self.net_sub_gen.load_model(idm_ckpt)

    def inference(self, lq, hq):
        """
        lq : 1 C H W
        """
        lq_clone = lq.clone().detach()
        hq = hq.to(self.execution_device)
        lq_clone = lq_clone.squeeze().permute(1, 2, 0).cpu().numpy()
        # print(lq_clone.shape)
        dt_boxes, rec_res = self.text_sys(lq_clone * 255)
        # print(dt_boxes)
        _, C, H, W = lq.shape
        # print(lq.shape)
        for i, (box, (text, score)) in enumerate(zip(dt_boxes, rec_res)):
            sub_lq = adjust_image_cv2(box, lq.clone().squeeze(0))
            C, subH, subW = sub_lq.shape
            # print(sub_lq.shape)
            # The gradients for the dtype float16 on CUDA may be inaccurate in the upsample operation when using modes ['linear', 'bilinear', 'bicubic', 'trilinear', 'area'].
            sub_lq = F.interpolate(sub_lq.unsqueeze(0), size=(128, 512), mode='bilinear').squeeze(0)
            # sub_lq, ratio, pad = resize_and_pad(sub_lq, (128, 512))
            # print(lq.shape)
            # print(sub_lq.shape)
            sub_sr = self.net_sub_gen.DiffTSR_sample(
                sub_lq.unsqueeze(0).to(self.execution_device)
            )
            # print(sub_mask.max())
            # print(sub_mask.min())
            # sub_sr = remove_padding(sub_sr, pad)
            sub_sr = F.interpolate(sub_sr, size=(subH, subW), mode='bilinear')
            sub_sr = restore_image_cv2(box, sub_sr.squeeze(0), ori_H=H, ori_W=W).unsqueeze(
                0
            )
            box = np.array(box, dtype=np.int32)

            # Create mask with the polygon
            pos = np.zeros((H, W, 1), dtype=np.uint8)  # h,w,c
            # print(pos.shape)
            cv2.fillPoly(pos, [box], (255, 255, 255))
            mask = pos
            mask =  torch.from_numpy(mask)
            mask = mask.permute(2,0,1).unsqueeze(0)
            mask = (mask>0).float().to(self.execution_device)
            # print(mask.shape)
            # print(hq.shape)
            # print(sub_sr.shape)
            hq = hq*(1-mask)+sub_sr
        return hq


class TextSR(nn.Module):
    def __init__(
        self,
        DiffTSR_yaml_config="./models/DiffTSR/DiffTSR_mask_config.yaml",
        target_timestep=150,
        ckpt_dir="./weights/stable-diffusion-3.5-medium",
        gradient_checkpointing=True,
        num_ctrl_layers=6,
    ):
        super().__init__()
        self.execution_device = torch.device("cuda")
        # initializing OCR Model
        # args = get_default_args_dict()
        # # print(args)
        # args["rec_char_dict_path"] = "./models/PaddleOCR2Pytorch/pytorchocr/utils/dict/ppocrv5_dict.txt"
        # args["det_model_path"] = (
        #     "./models/PaddleOCR2Pytorch/weights/ptocr_v5_server_det.pth"
        # )
        # args["det_yaml_path"] = (
        #     "./models/PaddleOCR2Pytorch/configs/det/PPOCRv5/PP-OCRv5_server_det.yml"
        # )
        # args["rec_image_shape"] = "3, 48, 320"
        # args["rec_model_path"] = (
        #     "./models/PaddleOCR2Pytorch/weights/ptocr_v5_server_rec.pth"
        # )
        # args["rec_yaml_path"] = (
        #     "./models/PaddleOCR2Pytorch/configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml"
        # )
        # args = argparse.Namespace(**args)
        # self.ocr = PaddleOCR(
        #     use_doc_orientation_classify=True,
        #     use_doc_unwarping=False,
        #     use_textline_orientation=True,
        #     device="cpu"
        # )
        # self.text_sys = TextSystem(args)
        # initializing phase 1
        DiffTSR_config = OmegaConf.load(DiffTSR_yaml_config)
        self.net_mask_gen = instantiate_from_config(DiffTSR_config.model)
        # self.net_mask_gen.load_model()

        # initializing phase 2
        self.net_pix2pix = SRCtrlnetPipeline(
            target_timestep=target_timestep,
            ckpt_dir=ckpt_dir,
            gradient_checkpointing=gradient_checkpointing,
            num_ctrl_layers=num_ctrl_layers,
        )

        # net_pix2pix.load_model(args.ckpt_path)

    def load_model(
        self,
        idm_ckpt,
        ctrl_ckpt="output/SR_hard/wSobel_from_20/checkpoints/model_Epoch_50.pkl",
    ):
        self.net_mask_gen.load_model()
        state_dict = torch.load(idm_ckpt, map_location="cpu")
        print(
            "Mask Gen IDM:",
            self.net_mask_gen.IDM_Unet.load_state_dict(
                state_dict["state_dict_dm"], strict=True
            ),
        )
        self.net_pix2pix.load_model(ctrl_ckpt)

    def set_eval(self):
        self.net_mask_gen.set_eval()
        self.net_pix2pix.set_eval()

    def to_device(self,device=None):
        self.execution_device = device if device is not None else self.execution_device
        self.net_mask_gen.to_device(device=device)
        self.net_pix2pix.to_device(device=device)

    def inference(self, lq):
        """
        lq : 1 C H W
        """
        lq_clone = lq.clone().detach()
        lq_clone = lq_clone.squeeze().permute(1, 2, 0).cpu().numpy()
        # print(lq_clone.shape)
        dt_boxes, rec_res = self.text_sys(lq_clone * 255)
        # print(dt_boxes)
        mask = torch.zeros_like(lq).to(self.execution_device)
        _, C, H, W = lq.shape
        # print(lq.shape)
        for i, (box, (text, score)) in enumerate(zip(dt_boxes, rec_res)):
            sub_lq = adjust_image(box, lq.clone().squeeze(0))
            C, subH, subW = sub_lq.shape
            if subH>subW and len(text)>1:
                sub_lq = torch.rot90(sub_lq, k=1, dims=[1, 2])
            # print(sub_lq.shape)
            # The gradients for the dtype float16 on CUDA may be inaccurate in the upsample operation when using modes ['linear', 'bilinear', 'bicubic', 'trilinear', 'area'].
            # sub_lq = torch.nn.functional.interpolate(sub_lq.unsqueeze(0), (128, 512), mode="nearest")
            sub_lq, ratio, pad = resize_and_pad(sub_lq, (128, 512))
            # print(lq.shape)
            # print(sub_lq.shape)
            sub_mask = self.net_mask_gen.inference(
                sub_lq.unsqueeze(0).to(self.execution_device), text
            )
            # print(sub_mask.max())
            # print(sub_mask.min())
            sub_mask = remove_padding(sub_mask, pad)
            # print(sub_mask.shape)
            # sub_mask = torch.nn.functional.interpolate(sub_mask, (subH, subW), mode="nearest").squeeze(0)
            # print(sub_lq.shape)
            if subH>subW and len(text)>1:
                sub_lq = torch.rot90(sub_lq, k=-1, dims=[1, 2])
            sub_mask = restore_image(
                box, sub_mask.squeeze(0), ori_H=H, ori_W=W
            ).unsqueeze(0)
            # sub_mask = torch.concat([sub_mask*3],dim=1)
            threshold = 0.8 * sub_mask.max()
            # threshold_layer = nn.Threshold(0.8 * sub_mask.max(), 0)  # 阈值为0.5，小于阈值的设为0
            # sub_mask = (threshold_layer(sub_mask)>0).float().to(self.execution_device)
            sub_mask = (sub_mask > threshold).any(dim=1, keepdim=True)
            sub_mask = (
                torch.cat([sub_mask] * 3, dim=1).float().to(self.execution_device)
            )
            mask += sub_mask
        SR = self.net_pix2pix(lq, mask)
        return SR, mask

    def inference_w_anno(self, lq, det_anno):
        """
        lq : 1 C H W
        """
        lq_clone = lq.clone().detach()
        lq_clone = lq_clone.squeeze().permute(1, 2, 0).cpu().numpy()
        # print(lq_clone.shape)
        # print(dt_boxes)
        mask = torch.zeros_like(lq).to(self.execution_device)
        _, C, H, W = lq.shape
        # print(lq.shape)
        for i, anno in enumerate(det_anno):
            text = anno["text"]
            quad = []
            for i in range(0, len(anno["segmentation"][0]), 2):
                if i + 1 < len(anno["segmentation"][0]):  # 确保有y坐标
                    point = [anno["segmentation"][0][i], anno["segmentation"][0][i + 1]]
                    quad.append(point)
            # print(text, quad)
            sub_lq = adjust_image(quad, lq.clone().squeeze(0))
            C, subH, subW = sub_lq.shape
            # print(sub_lq.shape)
            # The gradients for the dtype float16 on CUDA may be inaccurate in the upsample operation when using modes ['linear', 'bilinear', 'bicubic', 'trilinear', 'area'].
            # sub_lq = torch.nn.functional.interpolate(sub_lq.unsqueeze(0), (128, 512), mode="nearest")
            sub_lq, ratio, pad = resize_and_pad(sub_lq, (128, 512))
            # print(lq.shape)
            # print(sub_lq.shape)
            sub_mask = self.net_mask_gen.inference(
                sub_lq.unsqueeze(0).to(self.execution_device), text
            )
            # print(sub_mask.max())
            # print(sub_mask.min())
            sub_mask = remove_padding(sub_mask, pad)
            # print(sub_mask.shape)
            # sub_mask = torch.nn.functional.interpolate(sub_mask, (subH, subW), mode="nearest").squeeze(0)
            # print(sub_lq.shape)
            sub_mask = restore_image(
                quad, sub_mask.squeeze(0), ori_H=H, ori_W=W
            ).unsqueeze(0)
            # sub_mask = torch.concat([sub_mask*3],dim=1)
            threshold = 0.8 * sub_mask.max()
            # threshold_layer = nn.Threshold(0.8 * sub_mask.max(), 0)  # 阈值为0.5，小于阈值的设为0
            # sub_mask = (threshold_layer(sub_mask)>0).float().to(self.execution_device)
            sub_mask = (sub_mask > threshold).any(dim=1, keepdim=True)
            if sub_mask.shape[1] == 1:
                sub_mask = (
                    torch.cat([sub_mask] * 3, dim=1).float().to(self.execution_device)
                )
            mask += sub_mask
        SR = self.net_pix2pix(lq, mask)
        return SR, mask


from hi_sam.modeling.predictor import SamPredictor
from hi_sam.modeling.build import model_registry


class TextSR_hisam(TextSR):

    def __init__(
        self,
        DiffTSR_yaml_config="./models/DiffTSR/DiffTSR_mask_config.yaml",
        target_timestep=150,
        ckpt_dir="./weights/stable-diffusion-3.5-medium",
        gradient_checkpointing=True,
        num_ctrl_layers=6,
        model_type="vit_h",
        hi_sam_ckpt="weights/BTS-test-test_best.pth",
    ):
        super().__init__(
            DiffTSR_yaml_config,
            target_timestep,
            ckpt_dir,
            gradient_checkpointing,
            num_ctrl_layers,
        )
        hi_sam_args = {
            'model_type': model_type,
            "checkpoint": hi_sam_ckpt,
            "attn_layers": 1,
            "prompt_len": 12,
            "hier_det": False,
        }
        hi_sam_args = argparse.Namespace(**hi_sam_args)
        hisam = model_registry[model_type](hi_sam_args)
        hisam.eval()
        hisam.to(self.execution_device)
        self.predictor = SamPredictor(hisam)

    def load_model(self, idm_ckpt, ctrl_ckpt="output/SR_hard/wSobel_from_20/checkpoints/model_Epoch_50.pkl"):
        self.net_mask_gen.load_model(idm_ckpt)
        self.net_pix2pix.load_model(ctrl_ckpt)

    

    def set_eval(self):
        # self.net_mask_gen.set_eval()
        self.net_pix2pix.set_eval()

    def to_device(self):
        # self.net_mask_gen.to_device()
        self.net_pix2pix.to_device()

    def inference(self, lq):
        """
        lq : 1 C H W
        """
        lq_clone = lq.clone().detach()
        lq_clone = lq_clone.squeeze().permute(1, 2, 0).cpu().numpy()
        # print(lq_clone.shape)
        dt_boxes, rec_res = self.text_sys(lq_clone * 255)
        # print(dt_boxes)
        mask = torch.zeros_like(lq).to(self.execution_device)
        _, C, H, W = lq.shape
        # print(lq.shape)
        for i, (box, (text, score)) in enumerate(zip(dt_boxes, rec_res)):
            sub_lq = adjust_image(box, lq.clone().squeeze(0))
            C, subH, subW = sub_lq.shape
            # print(sub_lq.shape)
            # The gradients for the dtype float16 on CUDA may be inaccurate in the upsample operation when using modes ['linear', 'bilinear', 'bicubic', 'trilinear', 'area'].
            # sub_lq = torch.nn.functional.interpolate(sub_lq.unsqueeze(0), (128, 512), mode="nearest")
            sub_lq, ratio, pad = resize_and_pad(sub_lq, (128, 512))
            # print(lq.shape)
            # print(sub_lq.shape)
            sub_sr = self.net_mask_gen.DiffTSR_sample(
                sub_lq.unsqueeze(0).to(self.execution_device)
            )
            # print(sub_mask.max())
            # print(sub_mask.min())
            sub_sr = remove_padding(sub_sr, pad)
            # print(sub_sr.squeeze(0).permute(1,2,0).cpu().numpy().shape)
            self.predictor.set_image(sub_sr.squeeze(0).permute(1, 2, 0).cpu().numpy())
            _, sub_mask, _, _ = self.predictor.predict_torch(multimask_output=False)
            # print(sub_mask.shape)
            # sub_mask = torch.nn.functional.interpolate(sub_mask, (subH, subW), mode="nearest").squeeze(0)
            # print(sub_lq.shape)
            sub_mask = restore_image(
                box, sub_mask.squeeze(0), ori_H=H, ori_W=W
            ).unsqueeze(0)
            # sub_mask = torch.concat([sub_mask*3],dim=1)
            threshold = 0.8 * sub_mask.max()
            # threshold_layer = nn.Threshold(0.8 * sub_mask.max(), 0)  # 阈值为0.5，小于阈值的设为0
            # sub_mask = (threshold_layer(sub_mask)>0).float().to(self.execution_device)
            sub_mask = (sub_mask > threshold).any(dim=1, keepdim=True)
            sub_mask = (
                torch.cat([sub_mask] * 3, dim=1).float().to(self.execution_device)
            )
            mask += sub_mask
        SR = self.net_pix2pix(lq, mask)
        return SR, mask

class TextSR_2Branch(TextSR):

    def load_model(
        self,
        idm_ckpt,
        ctrl_ckpt="output/SR_hard/wSobel_from_20/checkpoints/model_Epoch_50.pkl",
    ):
        self.net_mask_gen.load_model(
            path="weights/DiffTSR/DiffTSR_new.ckpt",
            IDM_path=idm_ckpt,
            transformer_path="weights/DiffTSR/Bert_only.ckpt",
            is_training=False,
        )
        self.net_pix2pix.load_model(ctrl_ckpt)

    def inference(self, lq):
        """
        lq : 1 C H W
        """
        B, C, H, W = lq.shape
        lq = F.interpolate(lq, (H * 2, W * 2), mode="bilinear")
        lq_clone = lq.clone().detach()
        lq_clone = lq_clone.squeeze().permute(1, 2, 0).cpu().numpy()
        # print(lq_clone.shape)
        dt_boxes, rec_res = self.text_sys(lq_clone * 255)
        # print(dt_boxes)
        mask = torch.zeros_like(lq).to(self.execution_device)
        empty_canavas = torch.zeros_like(lq).to(self.execution_device)
        # print(lq.shape)
        for i, (box, (text, score)) in enumerate(zip(dt_boxes, rec_res)):
            sub_lq = adjust_image(box, lq.clone().squeeze(0))
            C, subH, subW = sub_lq.shape
            # print(sub_lq.shape)
            # The gradients for the dtype float16 on CUDA may be inaccurate in the upsample operation when using modes ['linear', 'bilinear', 'bicubic', 'trilinear', 'area'].
            # sub_lq = torch.nn.functional.interpolate(sub_lq.unsqueeze(0), (128, 512), mode="nearest")
            sub_lq, ratio, pad = resize_and_pad(sub_lq, (128, 512))
            # print(lq.shape)
            # print(sub_lq.shape)
            sub_mask, sub_sr = self.net_mask_gen.Baseline_sample(
                sub_lq.unsqueeze(0).to(self.execution_device), text
            )
            # print(sub_mask.max())
            # print(sub_mask.min())
            sub_mask = remove_padding(sub_mask, pad)
            sub_sr = remove_padding(sub_sr, pad)
            # print(sub_mask.shape)
            # sub_mask = torch.nn.functional.interpolate(sub_mask, (subH, subW), mode="nearest").squeeze(0)
            # print(sub_lq.shape)
            sub_mask = restore_image(
                box, sub_mask.squeeze(0), ori_H=H * 2, ori_W=W * 2
            ).unsqueeze(0)
            sub_sr = restore_image(box, sub_sr.squeeze(0), ori_H=H * 2, ori_W=W * 2)
            # sub_mask = torch.concat([sub_mask*3],dim=1)
            # threshold = 0
            # threshold_layer = nn.Threshold(0.8 * sub_mask.max(), 0)  # 阈值为0.5，小于阈值的设为0
            # sub_mask = (threshold_layer(sub_mask)>0).float().to(self.execution_device)
            # sub_mask = (sub_mask > threshold).any(dim=1, keepdim=True)
            if sub_mask.shape[1]==1:
                sub_mask = (
                    torch.cat([sub_mask] * 3, dim=1).float().to(self.execution_device)
                )
            mask += sub_mask
            empty_canavas += sub_sr
        subs = extract_overlapping_patches(
            torch.cat([lq.to(self.execution_device), mask], dim=0),
            patch_size=(H, W),
            overlap=(H // 2, W // 2),
        )
        sub_is,sub_ms = subs.chunk(2,dim=0)
        sub_SRs = []
        sub_masks = []
        for sub_i,sub_m in zip(sub_is,sub_ms):
            sub_SR = self.net_pix2pix(sub_i.unsqueeze(0), sub_m.unsqueeze(0),width=W,
                    height=H,)
            sub_SRs.append(sub_SR)
            sub_masks.append(sub_m.unsqueeze(0))
        sr_patches = torch.cat(sub_SRs, dim=0)
        mask_patches = torch.cat(sub_masks, dim=0)
        SR = combine_sr_patches_weighted(
            sr_patches,
            out_shape=(B, C, H * 2, W * 2),
            patch_size=(H, W),
            overlap=(H // 2, W // 2),
        )
        mask = combine_mask_patches(
            mask_patches,
            out_shape=(B, C, H * 2, W * 2),
            patch_size=(H, W),
            overlap=(H // 2, W // 2),
        )
        SR = F.interpolate(SR, (H, W), mode="bilinear")
        mask = F.interpolate(mask, (H, W), mode="bilinear")
        empty_canavas = F.interpolate(empty_canavas, (H, W), mode="bilinear")
        return SR, mask, empty_canavas

    def inference_w_anno(self, lq,det_anno):
        """
        lq : 1 C H W
        """
        lq_clone = lq.clone().detach()
        lq_clone = lq_clone.squeeze().permute(1, 2, 0).cpu().numpy()
        # print(lq_clone.shape)
        # print(dt_boxes)
        mask = torch.zeros_like(lq).to(self.execution_device)
        empty_canavas = torch.zeros_like(lq).to(self.execution_device)
        _, C, H, W = lq.shape
        # print(lq.shape)
        for i, anno in enumerate(det_anno):
            text = anno["text"]
            quad = []
            for i in range(0, len(anno["segmentation"][0]), 2):
                if i + 1 < len(anno["segmentation"][0]):  # 确保有y坐标
                    point = [anno["segmentation"][0][i], anno["segmentation"][0][i + 1]]
                    quad.append(point)
            # print(text, quad)
            sub_lq = adjust_image(quad, lq.clone().squeeze(0))
            C, subH, subW = sub_lq.shape
            # print(sub_lq.shape)
            # The gradients for the dtype float16 on CUDA may be inaccurate in the upsample operation when using modes ['linear', 'bilinear', 'bicubic', 'trilinear', 'area'].
            # sub_lq = torch.nn.functional.interpolate(sub_lq.unsqueeze(0), (128, 512), mode="nearest")
            sub_lq, ratio, pad = resize_and_pad(sub_lq, (128, 512))
            # print(lq.shape)
            # print(sub_lq.shape)
            sub_mask, sub_sr = self.net_mask_gen.Baseline_sample(
                sub_lq.unsqueeze(0).to(self.execution_device), text
            )
            # print(sub_mask.max())
            # print(sub_mask.min())
            sub_mask = remove_padding(sub_mask, pad)
            sub_sr = remove_padding(sub_sr,pad)
            # print(sub_mask.shape)
            # sub_mask = torch.nn.functional.interpolate(sub_mask, (subH, subW), mode="nearest").squeeze(0)
            # print(sub_lq.shape)
            sub_mask = restore_image(
                quad, sub_mask.squeeze(0), ori_H=H, ori_W=W
            ).unsqueeze(0)
            sub_sr = restore_image(quad, sub_sr.squeeze(0),ori_H=H,ori_W=W)
            # sub_mask = torch.concat([sub_mask*3],dim=1)
            # threshold = 0
            # threshold_layer = nn.Threshold(0.8 * sub_mask.max(), 0)  # 阈值为0.5，小于阈值的设为0
            # sub_mask = (threshold_layer(sub_mask)>0).float().to(self.execution_device)
            # sub_mask = (sub_mask > threshold).any(dim=1, keepdim=True)
            if sub_mask.shape[1] == 1:
                sub_mask = (
                    torch.cat([sub_mask] * 3, dim=1).float().to(self.execution_device)
                )
            mask += sub_mask
            empty_canavas += sub_sr
        SR = self.net_pix2pix(lq, maskwidth=W,
                    height=H,)
        return SR, mask,empty_canavas


class TextSR_2Out(TextSR_2Branch):
    def __init__(
        self,
        DiffTSR_yaml_config="./models/DiffTSR/DiffTSR_mask_config.yaml",
        target_timestep=150,
        ckpt_dir="./weights/stable-diffusion-3.5-medium",
        gradient_checkpointing=True,
        num_ctrl_layers=6,
    ):
        super().__init__(
            DiffTSR_yaml_config,
            target_timestep,
            ckpt_dir,
            gradient_checkpointing,
            num_ctrl_layers
        )
        self.ocr = None
        # self.ocr = PaddleOCR(
        #     use_doc_orientation_classify=True,
        #     use_doc_unwarping=False,
        #     use_textline_orientation=True,
        #     device="gpu"
        # )
        # self.modelBBox = TransformerOCR(use_new_bbox=True)
        # self.modelBBox.load_state_dict(torch.load('./weights/net_new_bbox.pth')['params'], strict=True)
        # self.modelBBox = self.modelBBox.eval().to(self.execution_device)

        # self.modelOCR = TransformerOCR()
        # self.modelOCR.load_state_dict(torch.load('./checkpoints/net_real_world_ocr.pth')['params'], strict=True)
        # self.modelOCR.eval()
        # self.modelOCR = self.modelOCR.to(self.execution_device)
    
    def load_model(
        self,
        idm_ckpt,
        ctrl_ckpt="output/SR_hard/wSobel_from_20/checkpoints/model_Epoch_50.pkl",
    ):
        self.net_mask_gen.load_model(
            path="weights/DiffTSR/DiffTSR_new.ckpt",
            IDM_path=idm_ckpt,
            transformer_path="weights/DiffTSR/Bert_only.ckpt",
            is_shape_diff=False,
        )
        self.net_pix2pix.load_model(ctrl_ckpt)

    def inference_wMaskSeg(self, lq):
        """
        lq : 1 C H W
        """
        B, C, H, W = lq.shape
        # lq = F.interpolate(lq, (H * 4, W * 4), mode="bilinear")
        lq_clone = lq.clone().detach()
        lq_clone = lq_clone.squeeze().permute(1, 2, 0).cpu().numpy()
        # print(lq_clone.shape)
        if self.ocr is None:
            self.ocr = PaddleOCR(
                use_doc_orientation_classify=True,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                device="gpu"
            )
        result = self.ocr.predict(lq_clone * 255)[0]
        img_angle = result["doc_preprocessor_res"]["angle"]
        if img_angle != -1:
            lq = torch.rot90(lq, k=img_angle // 90, dims=[2, 3])
            B, C, H, W = lq.shape
            # H = H // 4
            # W = W // 4
            # lq_clone = torch.rot90(lq_clone, k=-img_angle // 90, dims=[1, 0])
        lq = F.interpolate(lq, (H * 4, W * 4), mode="bilinear")
        dt_boxes = [box*4 for box in result["dt_polys"]]
        rec_res = result["rec_texts"]
        rec_scores = result["rec_scores"]
        try:
            textline_orientations = result["textline_orientation_angles"]
        except:
            print("Can't output textline orientations somehow")
            print(result)
            textline_orientations = [0]*len(rec_res)
        # print(dt_boxes)
        mask = torch.zeros_like(lq).to(self.execution_device)
        empty_canavas = torch.zeros_like(lq).to(self.execution_device)
        # print(lq.shape)
        iter = tqdm(range(len(dt_boxes)), desc="Textline Processing")
        for i, (box, text,score,orientation) in enumerate(zip(dt_boxes, rec_res,rec_scores, textline_orientations)):
            if score < 0.7:
                continue
            sub_lq = adjust_image(box, lq.clone().squeeze(0))
            if orientation==1:
                sub_lq = torch.rot90(sub_lq, k=2, dims=[1, 2])
            C, subH, subW = sub_lq.shape
            rot = False
            if subH/subW>=2 and len(text)>1:
                rot=True
                sub_lq = torch.rot90(sub_lq, k=1, dims=[1, 2])
                C, subH, subW = sub_lq.shape
            # print(sub_lq.shape)
            # The gradients for the dtype float16 on CUDA may be inaccurate in the upsample operation when using modes ['linear', 'bilinear', 'bicubic', 'trilinear', 'area'].
            # sub_lq = torch.nn.functional.interpolate(sub_lq.unsqueeze(0), (128, 512), mode="nearest")
            # print(text)
            # print("subW/subH",subW/subH)
            blocks_with_pos = split_long_image_tensor(sub_lq.unsqueeze(0),target_ratio=4,overlap_ratio=0)
            # text_blocks = split_text_annotation(text, blocks_with_pos, subW)
            sub_mask = []
            sub_sr = []

            for (block, pos) in blocks_with_pos:
                res = self.ocr.predict(block.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255)
                sub_rec_res  = res[0]["rec_texts"]
                sub_texts = ""
                for te in sub_rec_res:
                    sub_texts += te
                block, ratio, pad = resize_and_pad(block, (128, 512))

                block_mask, block_sr = self.net_mask_gen.Baseline_sample(
                    block.to(self.execution_device), sub_texts,disable_pbar=True
                )
                block_mask = remove_padding(block_mask, pad)
                block_sr = remove_padding(block_sr, pad)
                # print(ratio)
                sub_mask.append((block_mask, (int(pos[0] * ratio[0]),int(pos[1] * ratio[1]))))
                sub_sr.append((block_sr, (int(pos[0] * ratio[0]),int(pos[1] * ratio[1]))))

            sub_mask = merge_image_blocks(
                sub_mask,
                (1, C, int(subH * ratio[1]), int(subW * ratio[0])),
                blend_method="max",
            )
            sub_sr = merge_image_blocks(
                sub_sr,
                (1, C, int(subH * ratio[1]), int(subW * ratio[0])),
                blend_method="max",
            )
            if rot:
                C, subW, subH = sub_lq.shape
                sub_mask = torch.rot90(sub_mask, k=-1, dims=[2, 3])
                sub_sr = torch.rot90(sub_sr, k=-1, dims=[2, 3])
            sub_mask = torch.nn.functional.interpolate(sub_mask, (subH, subW), mode="nearest").squeeze(0)
            if orientation==1:
                sub_mask = torch.rot90(sub_mask, k=2, dims=[1, 2])
                sub_sr = torch.rot90(sub_sr, k=2, dims=[1, 2])
            sub_mask = restore_image(
                box, sub_mask.squeeze(0), ori_H=H * 4, ori_W=W * 4
            ).unsqueeze(0)
            sub_sr = restore_image(box, sub_sr.squeeze(0), ori_H=H * 4, ori_W=W * 4)
            if sub_mask.shape[1] == 1:
                sub_mask = (
                    torch.cat([sub_mask] * 3, dim=1).float().to(self.execution_device)
                )
            mask += sub_mask
            empty_canavas += sub_sr
            iter.update()

        # patch_size = (int(H * ratio) // 8 * 8, int(W * ratio) // 8 * 8)
        # overlap = (int(H * ratio) // 8 * 8 // 2, int(W * ratio) // 8 * 8 // 2)
        patch_size = (H,W)
        overlap = (H//2,W//2)
        subs = extract_overlapping_patches(
            torch.cat([lq.to(self.execution_device), mask], dim=0),
            patch_size=patch_size,
            overlap=overlap,
        )
        sub_is, sub_ms = subs.chunk(2, dim=0)
        sub_SRs = []
        iter = tqdm(range(sub_is.shape[0]), desc="Tiling")
        for sub_i, sub_m in zip(sub_is, sub_ms):
            sub_SR = self.net_pix2pix(sub_i.unsqueeze(0), sub_m.unsqueeze(0))
            sub_SRs.append(sub_SR)
            iter.update()
        sr_patches = torch.cat(sub_SRs, dim=0)
        SR = combine_sr_patches_weighted(
            sr_patches,
            out_shape=(B, C, H * 4, W * 4),
            patch_size=patch_size,
            overlap=overlap,
        )
        mask = combine_mask_patches(
            sub_ms,
            out_shape=(B, C, H * 4, W * 4),
            patch_size=patch_size,
            overlap=overlap,
        )
        SR = F.interpolate(SR, (H, W), mode="bilinear")
        mask = F.interpolate(mask, (H, W), mode="bilinear")
        empty_canavas = F.interpolate(empty_canavas, (H, W), mode="bilinear")
        if img_angle != -1:
            SR = torch.rot90(SR, k=-img_angle // 90, dims=[2, 3])
            mask = torch.rot90(mask, k=-img_angle // 90, dims=[2, 3])
            empty_canavas = torch.rot90(empty_canavas, k=-img_angle // 90, dims=[2, 3])
        return SR, mask, empty_canavas


    def inference_woPad(self, lq):
        """
        lq : 1 C H W
        """
        B, C, H, W = lq.shape
        lq = F.interpolate(lq, (H * 4, W * 4), mode="bilinear")
        lq_clone = lq.clone().detach()
        lq_clone = lq_clone.squeeze().permute(1, 2, 0).cpu().numpy()
        # print(lq_clone.shape)
        dt_boxes, rec_res = self.text_sys(lq_clone * 255)
        # print(dt_boxes)
        mask = torch.zeros_like(lq).to(self.execution_device)
        empty_canavas = torch.zeros_like(lq).to(self.execution_device)
        # print(lq.shape)
        for i, (box, (text, score)) in enumerate(zip(dt_boxes, rec_res)):
            rot = False
            sub_lq = adjust_image(box, lq.clone().squeeze(0))
            C, subH, subW = sub_lq.shape
            # print(sub_lq.shape)
            if subH/subW>=2 and len(text)>1:
                rot=True
                sub_lq = torch.rot90(sub_lq, k=1, dims=[1, 2])
                C, subH, subW = sub_lq.shape
            ratio = (128 / subH)
            # The gradients for the dtype float16 on CUDA may be inaccurate in the upsample operation when using modes ['linear', 'bilinear', 'bicubic', 'trilinear', 'area'].
            sub_lq = torch.nn.functional.interpolate(sub_lq.unsqueeze(0), (128, math.ceil(subW * ratio / 128) * 128), mode="nearest").squeeze(0)
            # print(sub_lq.shape)
            # sub_lq, ratio, pad = resize_and_pad(sub_lq, (128, 512))
            # print(lq.shape)
            # print(sub_lq.shape)
            sub_mask, sub_sr = self.net_mask_gen.Baseline_sample(
                sub_lq.unsqueeze(0).to(self.execution_device), text
            )
            if rot:
                C, subW, subH = sub_lq.shape
                sub_mask = torch.rot90(sub_mask, k=-1, dims=[2, 3])
                sub_sr = torch.rot90(sub_sr, k=-1, dims=[2, 3])
            sub_mask = torch.nn.functional.interpolate(sub_mask, (subH, subW), mode="nearest").squeeze(0)
            sub_sr = torch.nn.functional.interpolate(sub_sr, (subH, subW), mode="nearest").squeeze(0)
            # print(sub_lq.shape)
            sub_mask = restore_image(
                box, sub_mask.squeeze(0), ori_H=H * 4, ori_W=W * 4
            ).unsqueeze(0)
            sub_sr = restore_image(box, sub_sr.squeeze(0), ori_H=H * 4, ori_W=W * 4)
            # sub_mask = torch.concat([sub_mask*3],dim=1)
            # threshold = 0
            # threshold_layer = nn.Threshold(0.8 * sub_mask.max(), 0)  # 阈值为0.5，小于阈值的设为0
            # sub_mask = (threshold_layer(sub_mask)>0).float().to(self.execution_device)
            # sub_mask = (sub_mask > threshold).any(dim=1, keepdim=True)
            if sub_mask.shape[1]==1:
                sub_mask = (
                    torch.cat([sub_mask] * 3, dim=1).float().to(self.execution_device)
                )
            mask += sub_mask
            empty_canavas += sub_sr
        subs = extract_overlapping_patches(
            torch.cat([lq.to(self.execution_device), mask], dim=0),
            patch_size=(H, W),
            overlap=(H // 2, W // 2),
        )
        sub_is,sub_ms = subs.chunk(2,dim=0)
        sub_SRs = []
        sub_masks = []
        for sub_i,sub_m in zip(sub_is,sub_ms):
            sub_SR = self.net_pix2pix(sub_i.unsqueeze(0), sub_m.unsqueeze(0),width=W,
                    height=H,)
            sub_SRs.append(sub_SR)
            sub_masks.append(sub_m.unsqueeze(0))
        sr_patches = torch.cat(sub_SRs, dim=0)
        mask_patches = torch.cat(sub_masks, dim=0)
        SR = combine_sr_patches_weighted(
            sr_patches,
            out_shape=(B, C, H * 4, W * 4),
            patch_size=(H, W),
            overlap=(H // 2, W // 2),
        )
        mask = combine_mask_patches(
            mask_patches,
            out_shape=(B, C, H * 4, W * 4),
            patch_size=(H, W),
            overlap=(H // 2, W // 2),
        )
        SR = F.interpolate(SR, (H, W), mode="bilinear")
        mask = F.interpolate(mask, (H, W), mode="bilinear")
        empty_canavas = F.interpolate(empty_canavas, (H, W), mode="bilinear")
        return SR, mask, empty_canavas
    
    @torch.no_grad
    def inference_woPad(self, lq):
        """
        lq : 1 C H W
        """
        B, C, H, W = lq.shape
        lq_clone = lq.clone().detach()
        lq_clone = lq_clone.squeeze().permute(1, 2, 0).cpu().numpy()
        # print(lq_clone.shape)
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            device="cpu"
        )
        result = self.ocr.predict(lq_clone * 255)[0]
        img_angle = result["doc_preprocessor_res"]["angle"]
        if img_angle != -1:
            lq = torch.rot90(lq, k=-img_angle // 90, dims=[2, 3])
            B, C, H, W = lq.shape
            # H = H // 4
            # W = W // 4
            # lq_clone = torch.rot90(lq_clone, k=-img_angle // 90, dims=[1, 0])
        lq = F.interpolate(lq, (H * 4, W * 4), mode="bilinear")
        dt_boxes = [box*4 for box in result["dt_polys"]]
        rec_res = result["rec_texts"]
        try:
            textline_orientations = result["textline_orientation_angles"]
        except:
            print("Can't output textline orientations somehow")
            print(result)
            textline_orientations = [0]*len(rec_res)
        # print(dt_boxes)
        mask = torch.zeros_like(lq).to(self.execution_device)
        empty_canavas = torch.zeros_like(lq).to(self.execution_device)
        # print(lq.shape)
        for i, (box, text,orientation) in enumerate(zip(dt_boxes, rec_res, textline_orientations)):
            rot = False
            sub_lq = adjust_image(box, lq.clone().squeeze(0))
            if orientation==1:
                sub_lq = torch.rot90(sub_lq, k=2, dims=[1, 2])
            C, subH, subW = sub_lq.shape
            # print(sub_lq.shape)
            if subH/subW>=2 and len(text)>1:
                rot=True
                sub_lq = torch.rot90(sub_lq, k=1, dims=[1, 2])
                C, subH, subW = sub_lq.shape
            ratio = (128 / subH)
            # The gradients for the dtype float16 on CUDA may be inaccurate in the upsample operation when using modes ['linear', 'bilinear', 'bicubic', 'trilinear', 'area'].
            sub_lq = torch.nn.functional.interpolate(sub_lq.unsqueeze(0), (128, math.ceil(subW * ratio / 128) * 128), mode="nearest").squeeze(0)
            # print(sub_lq.shape)
            # sub_lq, ratio, pad = resize_and_pad(sub_lq, (128, 512))
            # print(lq.shape)
            # print(sub_lq.shape)
            sub_mask, sub_sr = self.net_mask_gen.Baseline_sample(
                sub_lq.unsqueeze(0).to(self.execution_device), text
            )
            if rot:
                C, subW, subH = sub_lq.shape
                sub_mask = torch.rot90(sub_mask, k=-1, dims=[2, 3])
                sub_sr = torch.rot90(sub_sr, k=-1, dims=[2, 3])
            sub_mask = torch.nn.functional.interpolate(sub_mask, (subH, subW), mode="nearest").squeeze(0)
            sub_sr = torch.nn.functional.interpolate(sub_sr, (subH, subW), mode="nearest").squeeze(0)
            if orientation==1:
                sub_mask = torch.rot90(sub_mask, k=2, dims=[1, 2])
                sub_sr = torch.rot90(sub_sr, k=2, dims=[1, 2])
            # print(sub_lq.shape)
            sub_mask = restore_image(
                box, sub_mask.squeeze(0), ori_H=H * 4, ori_W=W * 4
            ).unsqueeze(0)
            sub_sr = restore_image(box, sub_sr.squeeze(0), ori_H=H * 4, ori_W=W * 4)
            # sub_mask = torch.concat([sub_mask*3],dim=1)
            # threshold = 0
            # threshold_layer = nn.Threshold(0.8 * sub_mask.max(), 0)  # 阈值为0.5，小于阈值的设为0
            # sub_mask = (threshold_layer(sub_mask)>0).float().to(self.execution_device)
            # sub_mask = (sub_mask > threshold).any(dim=1, keepdim=True)
            if sub_mask.shape[1]==1:
                sub_mask = (
                    torch.cat([sub_mask] * 3, dim=1).float().to(self.execution_device)
                )
            try:
                mask += sub_mask
                empty_canavas += sub_sr
            except:
                print("lq",lq.shape)
                print("mask",mask.shape)
                print("sub mask",sub_mask.shape)
        subs = extract_overlapping_patches(
            torch.cat([lq.to(self.execution_device), mask], dim=0),
            patch_size=(H, W),
            overlap=(H // 2, W // 2),
        )
        sub_is,sub_ms = subs.chunk(2,dim=0)
        del subs,sub_sr,mask,sub_mask,lq_clone
        torch.cuda.empty_cache()
        # mask_patches = sub_ms
        # sr_patches = self.net_pix2pix(sub_is, sub_ms,width=W,
        #             height=H,)
        sub_SRs = []
        sub_masks = []
        for sub_i,sub_m in zip(sub_is,sub_ms):
            sub_SR = self.net_pix2pix(sub_i.unsqueeze(0), sub_m.unsqueeze(0),width=W,
                    height=H,)
            sub_SRs.append(sub_SR)
            sub_masks.append(sub_m.unsqueeze(0))
        sr_patches = torch.cat(sub_SRs, dim=0)
        mask_patches = torch.cat(sub_masks, dim=0)
        SR = combine_sr_patches_weighted(
            sr_patches,
            out_shape=(B, C, H * 4, W * 4),
            patch_size=(H, W),
            overlap=(H // 2, W // 2),
        )
        mask = combine_mask_patches(
            mask_patches,
            out_shape=(B, C, H * 4, W * 4),
            patch_size=(H, W),
            overlap=(H // 2, W // 2),
        )
        SR = F.interpolate(SR, (H, W), mode="bilinear")
        mask = F.interpolate(mask, (H, W), mode="bilinear")
        empty_canavas = F.interpolate(empty_canavas, (H, W), mode="bilinear")
        if img_angle != -1:
            SR = torch.rot90(SR, k=img_angle // 90, dims=[2, 3])
            mask = torch.rot90(mask, k=img_angle // 90, dims=[2, 3])
            empty_canavas = torch.rot90(empty_canavas, k=img_angle // 90, dims=[2, 3])
        return SR, mask, empty_canavas

class TextSR_2Outwskip(TextSR_2Out):

    def __init__(
        self,
        DiffTSR_yaml_config="./models/DiffTSR/DiffTSR_mask_config.yaml",
        target_timestep=150,
        ckpt_dir="./weights/stable-diffusion-3.5-medium",
        gradient_checkpointing=True,
        num_ctrl_layers=6,
    ):
        super().__init__(DiffTSR_yaml_config,
        target_timestep,
        ckpt_dir,
        gradient_checkpointing,
        num_ctrl_layers)
        # initializing phase 2
        self.net_pix2pix = SRCtrlnetSkipConnectionPipeline(
            target_timestep=target_timestep,
            ckpt_dir=ckpt_dir,
            gradient_checkpointing=gradient_checkpointing,
            num_ctrl_layers=num_ctrl_layers,
        )

        # net_pix2pix.load_model(args.ckpt_path)

    def load_model(
        self,
        idm_ckpt,
        ctrl_ckpt="output/SR_hard/wSobel_from_20/checkpoints/model_Epoch_50.pkl",
        enc_path="weights/DiffTSR/Bert_only.ckpt",
    ):
        self.net_mask_gen.load_model(
            path="weights/DiffTSR/DiffTSR_new.ckpt",
            IDM_path=idm_ckpt,
            transformer_path=enc_path,
            is_shape_diff=False,
        )
        self.net_pix2pix.load_model(ctrl_ckpt)

    @torch.no_grad()
    def inference(self, lq, anno=None, tiling=True):
        """
        统一的推理函数，可以处理两种情况：
        1. 使用内部OCR预测（anno=None）
        2. 使用提供的注释数据（anno不为None）
        
        参数:
        lq : 1 C H W - 输入的低质量图像
        anno : 可选，预先计算的OCR注释数据
        tiling : 是否使用图像分块处理，默认为True
        """
        B, C, H, W = lq.shape
        lq_clone = lq.clone().detach()
        lq_clone = lq_clone.squeeze().permute(1, 2, 0).cpu().numpy()
        
        # 根据是否提供anno决定使用OCR预测还是直接使用提供的注释
        if anno is None:
            # 使用OCR预测
            if self.ocr is None:
                self.ocr = PaddleOCR(
                    use_doc_orientation_classify=True,
                    use_doc_unwarping=False,
                    use_textline_orientation=True,
                    device="gpu"
                )
            result = self.ocr.predict(lq_clone * 255)[0]
            img_angle = result["doc_preprocessor_res"]["angle"]
        else:
            # 使用提供的注释
            result = anno
            img_angle = anno["doc_preprocessor_res"]["angle"]
        
        # 根据图像角度旋转
        if img_angle != -1:
            lq = torch.rot90(lq, k=img_angle // 90, dims=[2, 3])
            B, C, H, W = lq.shape
        
        # 根据tiling参数设置新的尺寸
        if tiling:
            new_H = H * 4
            new_W = W * 4
            lq = F.interpolate(lq, (new_H, new_W), mode="bilinear")
            dt_boxes = [np.array(box) * 4 for box in result["dt_polys"]]
        else:
            new_H = H
            new_W = W
            dt_boxes = [np.array(box) for box in result["dt_polys"]]
        
        rec_res = result["rec_texts"]
        rec_scores = result["rec_scores"]
        
        try:
            textline_orientations = result["textline_orientation_angles"]
        except:
            print("Can't output textline orientations somehow")
            print(result)
            textline_orientations = [0] * len(rec_res)
        
        mask = torch.zeros_like(lq).to(self.execution_device)
        empty_canavas = torch.zeros_like(lq).to(self.execution_device)
        
        # 处理每个文本框
        for i, (box, text, score, orientation) in enumerate(zip(dt_boxes, rec_res, rec_scores, textline_orientations)):
            # if score < 0.7:
            #     continue
            
            sub_lq = adjust_image(box, lq.clone().squeeze(0))
            if orientation == 1:
                sub_lq = torch.rot90(sub_lq, k=2, dims=[1, 2])
            
            C, subH, subW = sub_lq.shape
            ratio = 128 / min(subH, subW)
            sub_lq = torch.nn.functional.interpolate(
                sub_lq.unsqueeze(0), 
                (math.ceil(subH * ratio / 128) * 128, math.ceil(subW * ratio / 128) * 128), 
                mode="nearest"
            ).squeeze(0)
            
            sub_mask, sub_sr = self.net_mask_gen.Baseline_sample(
                sub_lq.unsqueeze(0).to(self.execution_device), text
            )
            
            sub_mask = torch.nn.functional.interpolate(sub_mask, (subH, subW), mode="nearest").squeeze(0)
            sub_sr = torch.nn.functional.interpolate(sub_sr, (subH, subW), mode="nearest").squeeze(0)
            
            if orientation == 1:
                sub_mask = torch.rot90(sub_mask, k=2, dims=[1, 2])
                sub_sr = torch.rot90(sub_sr, k=2, dims=[1, 2])
            
            # 处理背景颜色和阈值（来自inference_wanno）
            top_left = sub_mask[:, 0, 0]
            top_right = sub_mask[:, 0, -1]
            bottom_left = sub_mask[:, -1, 0]
            bottom_right = sub_mask[:, -1, -1]
            bg_color = ((top_left + top_right + bottom_left + bottom_right) / 4.0).unsqueeze(-1).unsqueeze(-1)
            
            m = sub_mask <= bg_color
            sub_mask[m] = 0
            # sub_mask = torch.sigmoid((sub_mask - sub_mask.max()/2) * 10000000)
            
            # m = sub_mask >= 0.8
            # sub_mask[m] = 1
            # sub_mask[~m] = 0
            
            sub_mask = restore_image(
                box, sub_mask.squeeze(0), ori_H=new_H, ori_W=new_W
            ).unsqueeze(0)
            
            sub_sr = restore_image(box, sub_sr.squeeze(0), ori_H=new_H, ori_W=new_W)
            
            if sub_mask.shape[1] == 1:
                sub_mask = torch.cat([sub_mask] * 3, dim=1).float().to(self.execution_device)
            
            mask = torch.maximum(mask, sub_mask)
            empty_canavas = torch.maximum(empty_canavas, sub_sr)
        
        # 根据tiling参数决定处理方式
        if tiling:
            subs = extract_overlapping_patches(
                torch.cat([lq.to(self.execution_device), mask], dim=0),
                patch_size=(H, W),
                overlap=(H // 2, W // 2),
            )
            sub_is, sub_ms = subs.chunk(2, dim=0)
            sub_SRs = []
            
            for sub_i, sub_m in zip(sub_is, sub_ms):
                sub_SR = self.net_pix2pix(
                    sub_i.unsqueeze(0), 
                    sub_m.unsqueeze(0),
                    width=W,
                    height=H,
                )
                sub_SRs.append(sub_SR)
            
            sr_patches = torch.cat(sub_SRs, dim=0)
            
            SR = combine_sr_patches_weighted(
                sr_patches,
                out_shape=(B, C, new_H, new_W),
                patch_size=(H, W),
                overlap=(H // 2, W // 2),
            )
            
        else:
            SR = self.net_pix2pix(lq.to(self.execution_device), mask, width=W, height=H)
        
        # 调整大小到原始尺寸
        SR = F.interpolate(SR, (H, W), mode="bilinear")
        mask = F.interpolate(mask, (H, W), mode="bilinear")
        empty_canavas = F.interpolate(empty_canavas, (H, W), mode="bilinear")
        
        # 根据原始角度旋转回来
        if img_angle != -1:
            SR = torch.rot90(SR, k=-img_angle // 90, dims=[2, 3])
            mask = torch.rot90(mask, k=-img_angle // 90, dims=[2, 3])
            empty_canavas = torch.rot90(empty_canavas, k=-img_angle // 90, dims=[2, 3])
        
        return SR, mask, empty_canavas


class TextSR_1Out(TextSR_2Outwskip):
    @torch.no_grad()  # 注意这里应该是装饰器，不是方法调用
    def inference_woPad(self, lq, tiling=True):
        """
        lq : 1 C H W
        tiling: 是否使用图像分块处理
        """
        B, C, H, W = lq.shape
        lq_clone = lq.clone().detach()
        lq_clone = lq_clone.squeeze().permute(1, 2, 0).cpu().numpy()

        if self.ocr is None:
            self.ocr = PaddleOCR(
                use_doc_orientation_classify=True,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                device="gpu"
            )
        result = self.ocr.predict(lq_clone * 255)[0]
        img_angle = result["doc_preprocessor_res"]["angle"]

        if img_angle != -1:
            lq = torch.rot90(lq, k=img_angle // 90, dims=[2, 3])
            B, C, H, W = lq.shape

        # 根据tiling参数决定是否进行上采样和边界框调整
        if tiling:
            new_H = H * 4
            new_W = W * 4
            lq = F.interpolate(lq, (new_H, new_W), mode="bilinear")
            dt_boxes = [box*4 for box in result["dt_polys"]]
        else:
            new_H = H
            new_W = W
            dt_boxes = result["dt_polys"]

        rec_res = result["rec_texts"]
        rec_scores = result["rec_scores"]
        try:
            textline_orientations = result["textline_orientation_angles"]
        except:
            print("Can't output textline orientations somehow")
            print(result)
            textline_orientations = [0]*len(rec_res)

        mask = torch.zeros_like(lq).to(self.execution_device)
        empty_canavas = torch.zeros_like(lq).to(self.execution_device)

        for i, (box, text, score, orientation) in enumerate(zip(dt_boxes, rec_res, rec_scores, textline_orientations)):
            if score < 0.7:
                continue
            sub_lq = adjust_image(box, lq.clone().squeeze(0))
            if orientation == 1:
                sub_lq = torch.rot90(sub_lq, k=2, dims=[1, 2])
            C, subH, subW = sub_lq.shape

            ratio = 128 / min(subH, subW)
            sub_lq = torch.nn.functional.interpolate(sub_lq.unsqueeze(0), (math.ceil(subH * ratio / 128) * 128, math.ceil(subW * ratio / 128) * 128), mode="nearest").squeeze(0)

            sub_mask = self.net_mask_gen.Baseline_sample(
                sub_lq.unsqueeze(0).to(self.execution_device), text
            )
            sub_mask = torch.nn.functional.interpolate(sub_mask, (subH, subW), mode="nearest").squeeze(0)
            if orientation == 1:
                sub_mask = torch.rot90(sub_mask, k=2, dims=[1, 2])

            # 添加背景颜色处理和阈值处理，与inference_wanno保持一致
            top_left     = sub_mask[:, 0, 0]
            top_right    = sub_mask[:, 0, -1]
            bottom_left  = sub_mask[:, -1, 0]
            bottom_right = sub_mask[:, -1, -1]
            bg_color = ((top_left + top_right + bottom_left + bottom_right) / 4.0).unsqueeze(-1).unsqueeze(-1)

            m = sub_mask <= bg_color
            sub_mask[m] = 0
            sub_mask = torch.sigmoid((sub_mask-sub_mask.max()/2)*10000000)

            m = sub_mask >= 0.5
            sub_mask[m] = 1
            sub_mask[~m] = 0

            sub_mask = restore_image(
                box, sub_mask.squeeze(0), ori_H=new_H, ori_W=new_W
            ).unsqueeze(0)

            if sub_mask.shape[1] == 1:
                sub_mask = torch.cat([sub_mask] * 3, dim=1).float().to(self.execution_device)

            mask = torch.maximum(mask, sub_mask)

        # 根据tiling参数决定处理方式
        if tiling:
            subs = extract_overlapping_patches(
                torch.cat([lq.to(self.execution_device), mask], dim=0),
                patch_size=(H, W),
                overlap=(H // 2, W // 2),
            )
            sub_is, sub_ms = subs.chunk(2, dim=0)
            sub_SRs = []
            sub_masks = []
            for sub_i, sub_m in zip(sub_is, sub_ms):
                sub_SR = self.net_pix2pix(sub_i.unsqueeze(0), sub_m.unsqueeze(0), width=W, height=H)
                sub_SRs.append(sub_SR)
                sub_masks.append(sub_m.unsqueeze(0))

            sr_patches = torch.cat(sub_SRs, dim=0)
            mask_patches = torch.cat(sub_masks, dim=0)

            SR = combine_sr_patches_weighted(
                sr_patches,
                out_shape=(B, C, new_H, new_W),
                patch_size=(H, W),
                overlap=(H // 2, W // 2),
            )
            mask = combine_mask_patches(
                mask_patches,
                out_shape=(B, C, new_H, new_W),
                patch_size=(H, W),
                overlap=(H // 2, W // 2),
            )
        else:
            SR = self.net_pix2pix(lq.to(self.execution_device), mask, width=W, height=H)

        SR = F.interpolate(SR, (H, W), mode="bilinear")
        mask = F.interpolate(mask, (H, W), mode="bilinear")
        empty_canavas = F.interpolate(empty_canavas, (H, W), mode="bilinear")

        if img_angle != -1:
            SR = torch.rot90(SR, k=-img_angle // 90, dims=[2, 3])
            mask = torch.rot90(mask, k=-img_angle // 90, dims=[2, 3])
            empty_canavas = torch.rot90(empty_canavas, k=-img_angle // 90, dims=[2, 3])

        return SR, mask, empty_canavas

    
    @torch.no_grad()
    def inference_wanno(self, lq, anno, tiling=True):
        """
        lq : 1 C H W
        """
        B, C, H, W = lq.shape
        lq_clone = lq.clone().detach()
        lq_clone = lq_clone.squeeze().permute(1, 2, 0).cpu().numpy()
        # print(lq_clone.shape)
        img_angle = anno["doc_preprocessor_res"]["angle"]
        if img_angle != -1:
            lq = torch.rot90(lq, k=img_angle // 90, dims=[2, 3])
            B, C, H, W = lq.shape
            # H = H // 4
            # W = W // 4
            # lq_clone = torch.rot90(lq_clone, k=-img_angle // 90, dims=[1, 0])
        if tiling:
            lq = F.interpolate(lq, (new_H, new_W), mode="bilinear")
            dt_boxes = [np.array(box) * 4 for box in anno["dt_polys"]]
        else:
            dt_boxes = [np.array(box) for box in anno["dt_polys"]]
        if tiling:
            new_H = H * 4
            new_W = W * 4
        else:
            new_H = H
            new_W = W
        rec_res = anno["rec_texts"]
        rec_scores = anno["rec_scores"]
        try:
            textline_orientations = anno["textline_orientation_angles"]
        except:
            print("Can't output textline orientations somehow")
            print(anno)
            textline_orientations = [0]*len(rec_res)
        # print(dt_boxes)
        mask = torch.zeros_like(lq).to(self.execution_device)
        # print(lq.shape)
        for i, (box, text,score,orientation) in enumerate(zip(dt_boxes, rec_res,rec_scores, textline_orientations)):
            if score < 0.7:
                continue
            sub_lq = adjust_image(box, lq.clone().squeeze(0))
            if orientation==1:
                sub_lq = torch.rot90(sub_lq, k=2, dims=[1, 2])
            C, subH, subW = sub_lq.shape
            # print(sub_lq.shape)
            ratio = 128 / min(subH,subW)
            # The gradients for the dtype float16 on CUDA may be inaccurate in the upsample operation when using modes ['linear', 'bilinear', 'bicubic', 'trilinear', 'area'].
            sub_lq = torch.nn.functional.interpolate(sub_lq.unsqueeze(0), (math.ceil(subH * ratio / 128) * 128, math.ceil(subW * ratio / 128) * 128), mode="nearest").squeeze(0)

            sub_mask = self.net_mask_gen.Baseline_sample(
                sub_lq.unsqueeze(0).to(self.execution_device), text
            )
            sub_mask = torch.nn.functional.interpolate(sub_mask, (subH, subW), mode="nearest").squeeze(0)
            if orientation==1:
                sub_mask = torch.rot90(sub_mask, k=2, dims=[1, 2])
            # print(sub_lq.shape)
              # 1. 提取4角像素点的值，shape = [ 3 h w]
            top_left     = sub_mask[ :, 0, 0]
            top_right    = sub_mask[:, 0, -1]
            bottom_left  = sub_mask[:, -1, 0]
            bottom_right = sub_mask[:, -1, -1]
            bg_color = ((top_left + top_right + bottom_left + bottom_right) / 4.0).unsqueeze(-1).unsqueeze(-1)
            # print(bg_color.shape)
            # print(sub_mask.shape)
            # print((sub_mask<=bg_color).shape)
            # print(sub_mask.shape)
            m = sub_mask<=bg_color
            sub_mask[m] = 0
            # print("Max",sub_mask.max())
            # print("Min",sub_mask.min())
            # print("Mean",sub_mask.mean())
            # sub_mask = torch.sigmoid((sub_mask-sub_mask.max()/2)*10000000)
            # kernel_size = 5
            # sub_mask = dilation(sub_mask.unsqueeze(0), kernel=torch.ones(kernel_size, kernel_size).to(self.execution_device))
            # sub_mask = torch.sigmoid((sub_mask-sub_mask.max()/2)*10000000)
            # m = sub_mask>=0.5
            # sub_mask[m] = 1
            # sub_mask[~m] = 0
            # print("Max",sub_mask.max())
            # print("Min",sub_mask.min())
            # print("Mean",sub_mask.mean())
            sub_mask = restore_image(
                box, sub_mask.squeeze(0), ori_H=new_H, ori_W=new_W
            ).unsqueeze(0)
            # sub_mask = torch.concat([sub_mask*3],dim=1)
            # threshold = 0
            # threshold_layer = nn.Threshold(0.8 * sub_mask.max(), 0)  # 阈值为0.5，小于阈值的设为0
            # sub_mask = (threshold_layer(sub_mask)>0).float().to(self.execution_device)
            # sub_mask = (sub_mask > threshold).any(dim=1, keepdim=True)
            if sub_mask.shape[1]==1:
                sub_mask = (
                    torch.cat([sub_mask] * 3, dim=1).float().to(self.execution_device)
                )
            mask = torch.maximum(mask, sub_mask)
        
        if tiling:
            subs = extract_overlapping_patches(
                torch.cat([lq.to(self.execution_device), mask], dim=0),
                patch_size=(H, W),
                overlap=(H // 2, W // 2),
            )
            sub_is,sub_ms = subs.chunk(2,dim=0)
            sub_SRs = []
            for sub_i,sub_m in zip(sub_is,sub_ms):
                sub_SR = self.net_pix2pix(sub_i.unsqueeze(0), sub_m.unsqueeze(0),width=W,
                        height=H,)
                sub_SRs.append(sub_SR)
            sr_patches = torch.cat(sub_SRs, dim=0)
            # mask_patches = sub_ms
            # sr_patches = self.net_pix2pix(sub_is, sub_ms,width=W,
            #             height=H,)
            # sr_patches = torch.cat(sub_SRs, dim=0)
            # mask_patches = torch.cat(sub_masks, dim=0)
            SR = combine_sr_patches_weighted(
                sr_patches,
                out_shape=(B, C, new_H, new_W),
                patch_size=(H, W),
                overlap=(H // 2, W // 2),
            )
        else:
            SR = self.net_pix2pix(lq.to(self.execution_device), mask, width=W,
                    height=H,)
        SR = F.interpolate(SR, (H, W), mode="bilinear")
        mask = F.interpolate(mask, (H, W), mode="bilinear")
        
        torch.set_printoptions(threshold=np.inf)
        # print(mask)
        if img_angle != -1:
            SR = torch.rot90(SR, k=-img_angle // 90, dims=[2, 3])
            mask = torch.rot90(mask, k=-img_angle // 90, dims=[2, 3])
        return SR, mask

# 文本超分辨率输出类
from utils.ocr_util import randomly_replace_chars_enhanced
class TextSR_multiscale(TextSR_2Outwskip):
    @torch.no_grad()
    def inference_woPad(self, lq):
        """
        lq : 1 C H W
        """
        B, C, H, W = lq.shape
        lq_clone = lq.clone().detach()
        lq_clone = lq_clone.squeeze().permute(1, 2, 0).cpu().numpy()
        # print(lq_clone.shape)
        if self.ocr is None:
            self.ocr = PaddleOCR(
                use_doc_orientation_classify=True,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                device="gpu"
            )
        result = self.ocr.predict(lq_clone * 255)[0]
        img_angle = result["doc_preprocessor_res"]["angle"]
        if img_angle != -1:
            lq = torch.rot90(lq, k=img_angle // 90, dims=[2, 3])
            B, C, H, W = lq.shape
            # H = H // 4
            # W = W // 4
            # lq_clone = torch.rot90(lq_clone, k=-img_angle // 90, dims=[1, 0])
        lq = F.interpolate(lq, (H * 4, W * 4), mode="bilinear")
        dt_boxes = [box*4 for box in result["dt_polys"]]
        rec_res = result["rec_texts"]
        try:
            rec_scores = result["rec_scores"]
        except:
            rec_scores = [1.0]*(len(rec_res))
        if len(rec_res) > len(rec_scores):
            rec_scores += [1.0]*(len(rec_res)-len(rec_scores))
        try:
            textline_orientations = result["textline_orientation_angles"]
        except:
            print("Can't output textline orientations somehow")
            print(result)
            textline_orientations = [0]*len(rec_res)
        # print(dt_boxes)
        mask = torch.zeros_like(lq).to(self.execution_device)
        empty_canavas = torch.zeros_like(lq).to(self.execution_device)
        # print(lq.shape)
        for i, (box, text,score,orientation) in enumerate(zip(dt_boxes, rec_res,rec_scores, textline_orientations)):
            if score < 0.7:
                continue
            sub_lq = adjust_image(box, lq.clone().squeeze(0))
            if orientation==1:
                sub_lq = torch.rot90(sub_lq, k=2, dims=[1, 2])
            C, subH, subW = sub_lq.shape
            # print(sub_lq.shape)
            ratio = 128 / min(subH,subW)
            # The gradients for the dtype float16 on CUDA may be inaccurate in the upsample operation when using modes ['linear', 'bilinear', 'bicubic', 'trilinear', 'area'].
            sub_lq = torch.nn.functional.interpolate(sub_lq.unsqueeze(0), (math.ceil(subH * ratio / 128) * 128, math.ceil(subW * ratio / 128) * 128), mode="nearest").squeeze(0)

            sub_mask, sub_sr = self.net_mask_gen.Baseline_sample(
                sub_lq.unsqueeze(0).to(self.execution_device), text
            )
            sub_mask = torch.nn.functional.interpolate(sub_mask, (subH, subW), mode="nearest").squeeze(0)
            sub_sr = torch.nn.functional.interpolate(sub_sr, (subH, subW), mode="nearest").squeeze(0)
            if orientation==1:
                sub_mask = torch.rot90(sub_mask, k=2, dims=[1, 2])
                sub_sr = torch.rot90(sub_sr, k=2, dims=[1, 2])
            # print(sub_lq.shape)
            sub_mask = restore_image(
                box, sub_mask.squeeze(0), ori_H=H * 4, ori_W=W * 4
            ).unsqueeze(0)
            sub_sr = restore_image(box, sub_sr.squeeze(0), ori_H=H * 4, ori_W=W * 4)
            # sub_mask = torch.concat([sub_mask*3],dim=1)
            # threshold = 0
            # threshold_layer = nn.Threshold(0.8 * sub_mask.max(), 0)  # 阈值为0.5，小于阈值的设为0
            # sub_mask = (threshold_layer(sub_mask)>0).float().to(self.execution_device)
            # sub_mask = (sub_mask > threshold).any(dim=1, keepdim=True)
            if sub_mask.shape[1]==1:
                sub_mask = (
                    torch.cat([sub_mask] * 3, dim=1).float().to(self.execution_device)
                )
            mask = torch.maximum(mask, sub_mask)
            empty_canavas = torch.maximum(empty_canavas, sub_sr)
        subs = extract_overlapping_patches(
            torch.cat([lq.to(self.execution_device), mask], dim=0),
            patch_size=(H, W),
            overlap=(H // 2, W // 2),
        )
        sub_is,sub_ms = subs.chunk(2,dim=0)
        sub_SRs = []
        sub_masks = []
        for sub_i,sub_m in zip(sub_is,sub_ms):
            sub_SR = self.net_pix2pix(sub_i.unsqueeze(0), sub_m.unsqueeze(0),width=W,
                    height=H,)
            sub_SRs.append(sub_SR)
            sub_masks.append(sub_m.unsqueeze(0))
        sr_patches = torch.cat(sub_SRs, dim=0)
        mask_patches = torch.cat(sub_masks, dim=0)
        # mask_patches = sub_ms
        # sr_patches = self.net_pix2pix(sub_is, sub_ms,width=W,
        #             height=H,)
        # sr_patches = torch.cat(sub_SRs, dim=0)
        # mask_patches = torch.cat(sub_masks, dim=0)
        SR = combine_sr_patches_weighted(
            sr_patches,
            out_shape=(B, C, H * 4, W * 4),
            patch_size=(H, W),
            overlap=(H // 2, W // 2),
        )
        mask = combine_mask_patches(
            mask_patches,
            out_shape=(B, C, H * 4, W * 4),
            patch_size=(H, W),
            overlap=(H // 2, W // 2),
        )
        SR = F.interpolate(SR, (H, W), mode="bilinear")
        mask = F.interpolate(mask, (H, W), mode="bilinear")
        empty_canavas = F.interpolate(empty_canavas, (H, W), mode="bilinear")
        if img_angle != -1:
            SR = torch.rot90(SR, k=-img_angle // 90, dims=[2, 3])
            mask = torch.rot90(mask, k=-img_angle // 90, dims=[2, 3])
            empty_canavas = torch.rot90(empty_canavas, k=-img_angle // 90, dims=[2, 3])
        return SR, mask, empty_canavas
    
    @torch.no_grad()
    def inference_wanno(self, lq, anno, tiling=True):
        """
        lq : 1 C H W
        """
        B, C, H, W = lq.shape
        lq_clone = lq.clone().detach()
        lq_clone = lq_clone.squeeze().permute(1, 2, 0).cpu().numpy()
        # print(lq_clone.shape)
        img_angle = anno["doc_preprocessor_res"]["angle"]
        if img_angle != -1:
            lq = torch.rot90(lq, k=img_angle // 90, dims=[2, 3])
            B, C, H, W = lq.shape
            # H = H // 4
            # W = W // 4
            # lq_clone = torch.rot90(lq_clone, k=-img_angle // 90, dims=[1, 0])
        if tiling:
            new_H = H * 4
            new_W = W * 4
        else:
            new_H = H
            new_W = W
        if tiling:
            lq = F.interpolate(lq, (new_H, new_W), mode="bilinear")
            dt_boxes = [np.array(box) * 4 for box in anno["dt_polys"]]
        else:
            dt_boxes = [np.array(box) for box in anno["dt_polys"]]
        rec_res = anno["rec_texts"]
        try:
            rec_scores = anno["rec_scores"]
        except:
            rec_scores = [1.0]*(len(rec_res))
        if len(rec_res) > len(rec_scores):
            rec_scores += [1.0]*(len(rec_res)-len(rec_scores))
        try:
            textline_orientations = anno["textline_orientation_angles"]
        except:
            print("Can't output textline orientations somehow")
            print(anno)
            textline_orientations = [0]*len(rec_res)
        if len(rec_res) > len(textline_orientations):
            textline_orientations += [0]*(len(rec_res)-len(textline_orientations))
        # print(dt_boxes)
        mask = torch.zeros_like(lq).to(self.execution_device)
        empty_canavas = torch.zeros_like(lq).to(self.execution_device)
        # print(lq.shape)
        for i, (box, text,score,orientation) in enumerate(zip(dt_boxes, rec_res,rec_scores, textline_orientations)):
            # if score < 0.7:
            #     continue
            # print(text,box)
            sub_lq = adjust_image_cv2(box, lq.clone().squeeze(0))
            # self.net_mask_gen.set_image(sub_lq.permute(1, 2, 0).cpu().numpy())
            # _, sub_mask, _, _ = self.predictor.predict_torch(multimask_output=False)
            if orientation==1:
                sub_lq = torch.rot90(sub_lq, k=2, dims=[1, 2])
            C, subH, subW = sub_lq.shape
            # print(sub_lq.shape)
            ratio = 128 / min(subH,subW)
            # The gradients for the dtype float16 on CUDA may be inaccurate in the upsample operation when using modes ['linear', 'bilinear', 'bicubic', 'trilinear', 'area'].
            sub_lq = torch.nn.functional.interpolate(sub_lq.unsqueeze(0), (math.ceil(subH * ratio / 128) * 128, math.ceil(subW * ratio / 128) * 128), mode="nearest").squeeze(0)
            # text = randomly_replace_chars_enhanced(text,replacement_rate=0.2)
            sub_mask, sub_sr = self.net_mask_gen.Baseline_sample(
                sub_lq.unsqueeze(0).to(self.execution_device), text
            )
            sub_mask = torch.nn.functional.interpolate(sub_mask, (subH, subW), mode="nearest").squeeze(0)
            sub_sr = torch.nn.functional.interpolate(sub_sr, (subH, subW), mode="nearest").squeeze(0)
            if orientation==1:
                sub_mask = torch.rot90(sub_mask, k=2, dims=[1, 2])
                sub_sr = torch.rot90(sub_sr, k=2, dims=[1, 2])
            # print(sub_lq.shape)
            #   1. 提取4角像素点的值，shape = [ 3 h w]
            top_left     = sub_mask[ :, 0, 0]
            top_right    = sub_mask[:, 0, -1]
            bottom_left  = sub_mask[:, -1, 0]
            bottom_right = sub_mask[:, -1, -1]
            bg_color = ((top_left + top_right + bottom_left + bottom_right) / 4.0).unsqueeze(-1).unsqueeze(-1)
            # print(bg_color.shape)
            # print(sub_mask.shape)
            # print((sub_mask<=bg_color).shape)
            # print(sub_mask.shape)
            m = sub_mask<=bg_color
            sub_mask[m] = 0
            # print("Max",sub_mask.max())
            # print("Min",sub_mask.min())
            # print("Mean",sub_mask.mean())
            # sub_mask = torch.sigmoid((sub_mask-sub_mask.max()/2)*10000000)
            # kernel_size = 5
            # sub_mask = dilation(sub_mask.unsqueeze(0), kernel=torch.ones(kernel_size, kernel_size).to(self.execution_device))
            # sub_mask = torch.sigmoid((sub_mask-sub_mask.max()/2)*10000000)
            # m = sub_mask>=0.5
            # sub_mask[m] = 1
            # sub_mask[~m] = 0
            # print("Max",sub_mask.max())
            # print("Min",sub_mask.min())
            # print("Mean",sub_mask.mean())
            sub_mask = restore_image_cv2(
                box, sub_mask.squeeze(0), ori_H=new_H, ori_W=new_W
            ).unsqueeze(0)
            sub_sr = restore_image_cv2(box, sub_sr.squeeze(0), ori_H=new_H, ori_W=new_W)
            # sub_mask = torch.concat([sub_mask*3],dim=1)
            # threshold = 0
            # threshold_layer = nn.Threshold(0.8 * sub_mask.max(), 0)  # 阈值为0.5，小于阈值的设为0
            # sub_mask = (threshold_layer(sub_mask)>0).float().to(self.execution_device)
            # sub_mask = (sub_mask > threshold).any(dim=1, keepdim=True)
            if sub_mask.shape[1]==1:
                sub_mask = (
                    torch.cat([sub_mask] * 3, dim=1).float().to(self.execution_device)
                )
            mask = torch.maximum(mask, sub_mask)
            empty_canavas = torch.maximum(empty_canavas, sub_sr)
        
        
        if tiling:
            subs = extract_overlapping_patches(
                torch.cat([lq.to(self.execution_device), mask], dim=0),
                patch_size=(H, W),
                overlap=(H // 2, W // 2),
            )
            sub_is,sub_ms = subs.chunk(2,dim=0)
            sub_SRs = []
            for sub_i,sub_m in zip(sub_is,sub_ms):
                sub_SR = self.net_pix2pix(sub_i.unsqueeze(0), sub_m.unsqueeze(0),width=W,
                        height=H,)
                sub_SRs.append(sub_SR)
            sr_patches = torch.cat(sub_SRs, dim=0)
            # mask_patches = sub_ms
            # sr_patches = self.net_pix2pix(sub_is, sub_ms,width=W,
            #             height=H,)
            # sr_patches = torch.cat(sub_SRs, dim=0)
            # mask_patches = torch.cat(sub_masks, dim=0)
            SR = combine_sr_patches_weighted(
                sr_patches,
                out_shape=(B, C, new_H, new_W),
                patch_size=(H, W),
                overlap=(H // 2, W // 2),
            )
        else:
            SR = self.net_pix2pix(lq.to(self.execution_device), mask, width=W,
                    height=H,)
        SR = F.interpolate(SR, (H, W), mode="bilinear")
        mask = F.interpolate(mask, (H, W), mode="bilinear")
        
        torch.set_printoptions(threshold=np.inf)
        # print(mask)
        empty_canavas = F.interpolate(empty_canavas, (H, W), mode="bilinear")
        if img_angle != -1:
            SR = torch.rot90(SR, k=-img_angle // 90, dims=[2, 3])
            mask = torch.rot90(mask, k=-img_angle // 90, dims=[2, 3])
            empty_canavas = torch.rot90(empty_canavas, k=-img_angle // 90, dims=[2, 3])
        return SR, mask, empty_canavas
