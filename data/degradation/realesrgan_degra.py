import glob
import math
import random

import cv2
import numpy as np
import torch
from src.datasets.degradation.degradation import (circular_lowpass_kernel,
                                                  random_add_gaussian_noise_pt,
                                                  random_add_poisson_noise_pt,
                                                  random_mixed_kernels)
from src.datasets.degradation.file import opt_parse_yaml
from src.datasets.degradation.image import (DiffJPEG, GaussianBlur,
                                            JEPG_degradation, USMSharp, augment,
                                            center_crop_arr, color_jitter_pt,
                                            filter2D, img2tensor,
                                            multi_scale_crop, random_crop_arr,
                                            resize_degradation)
from torch.nn import functional as F
from torchvision.transforms.functional import rgb_to_grayscale
from src.datasets.degradation.text import TextDraw


class RealESRGAN_Degra(object):
    """
    # TODO: add comment
    """

    def __init__(self, degradation_set_path):
        self.opt = opt_parse_yaml(degradation_set_path)

        # default CCSR set
        self.out_size = self.opt['out_size'] if 'out_size' in self.opt.keys(
        ) else 512
        self.use_hflip = self.opt['use_hflip'] if 'use_hflip' in self.opt.keys(
        ) else False
        self.use_rot = self.opt['use_rot'] if 'use_rot' in self.opt.keys(
        ) else False
        self.crop_type = self.opt['crop_type'] if 'crop_type' in self.opt.keys(
        ) else 'multi_scale'
        assert self.crop_type in [
            "center", "random", "multi_scale", "none"], f"invalid crop type: {self.crop_type}"
        self.color_jitter_prob = self.opt['color_jitter_prob'] if 'color_jitter_prob' in self.opt.keys(
        ) else 0.0
        self.gray_prob = self.opt['gray_prob'] if 'gray_prob' in self.opt.keys(
        ) else 0.0

        self.first_resize_prob = self.opt['first_resize_prob'] if 'first_resize_prob' in self.opt.keys() else [
            0.2, 0.7, 0.1]  # up, down, keep
        self.first_resize_range = self.opt['first_resize_range'] if 'first_resize_range' in self.opt.keys() else [
            0.15, 1.5]
        self.first_add_noise_prob = self.opt['first_add_noise_prob'] if 'first_add_noise_prob' in self.opt.keys(
        ) else 0.5
        self.first_gaussian_noise_prob = self.opt['first_gaussian_noise_prob'] if 'first_gaussian_noise_prob' in self.opt.keys(
        ) else 0.5
        self.first_noise_range = self.opt['first_noise_range'] if 'first_noise_range' in self.opt.keys() else [
            1, 30]
        self.first_poisson_scale_range = self.opt['first_poisson_scale_range'] if 'first_poisson_scale_range' in self.opt.keys() else [
            0.05, 3]
        self.first_gray_noise_prob = self.opt['first_gray_noise_prob'] if 'first_gray_noise_prob' in self.opt.keys(
        ) else 0.4
        self.first_jpeg_range = self.opt['first_jpeg_range'] if 'first_jpeg_range' in self.opt.keys() else [
            30, 95]

        # the second degradation process
        self.second_blur_prob = self.opt['second_blur_prob'] if 'second_blur_prob' in self.opt.keys(
        ) else 0.8
        self.second_resize_prob = self.opt['second_resize_prob'] if 'second_resize_prob' in self.opt.keys() else [
            0.3, 0.4, 0.3]  # up, down, keep
        self.second_resize_range = self.opt['second_resize_range'] if 'second_resize_range' in self.opt.keys() else [
            0.3, 1.2]
        self.second_add_noise_prob = self.opt['second_add_noise_prob'] if 'second_add_noise_prob' in self.opt.keys(
        ) else 0.5
        self.second_gaussian_noise_prob = self.opt['second_gaussian_noise_prob'] if 'second_gaussian_noise_prob' in self.opt.keys(
        ) else 0.5
        self.second_noise_range = self.opt['second_noise_range'] if 'second_noise_range' in self.opt.keys() else [
            1, 25]
        self.second_poisson_scale_range = self.opt['second_poisson_scale_range'] if 'second_poisson_scale_range' in self.opt.keys() else [
            0.05, 2.5]
        self.second_gray_noise_prob = self.opt['second_gray_noise_prob'] if 'second_gray_noise_prob' in self.opt.keys(
        ) else 0.4
        self.second_jpeg_range = self.opt['second_jpeg_range'] if 'second_jpeg_range' in self.opt.keys() else [
            30, 95]

        optk = self.opt['kernel_info'] if 'kernel_info' in self.opt.keys(
        ) else None
        # blur settings for the first degradation
        self.first_blur_kernel_size = optk['first_blur_kernel_size'] if 'first_blur_kernel_size' in optk.keys(
        ) else 21
        self.first_kernel_list = optk['first_kernel_list'] if 'first_kernel_list' in optk.keys(
        ) else ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.first_kernel_prob = optk['first_kernel_prob'] if 'first_kernel_prob' in optk.keys(
        ) else [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.first_blur_sigma = optk['first_blur_sigma'] if 'first_blur_sigma' in optk.keys(
        ) else [0.2, 3]
        self.first_betag_range = optk['first_betag_range'] if 'first_betag_range' in optk.keys(
        ) else [0.5, 4]
        self.first_betap_range = optk['first_betap_range'] if 'first_betap_range' in optk.keys(
        ) else [1, 2]
        self.first_sinc_prob = optk['first_sinc_prob'] if 'first_sinc_prob' in optk.keys(
        ) else 0.1

        # blur settings for the second degradation
        self.second_blur_kernel_size = optk['second_blur_kernel_size'] if 'second_blur_kernel_size' in optk.keys(
        ) else 11
        self.second_kernel_list = optk['second_kernel_list'] if 'second_kernel_list' in optk.keys(
        ) else ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.second_kernel_prob = optk['second_kernel_prob'] if 'second_kernel_prob' in optk.keys(
        ) else [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.second_blur_sigma = optk['second_blur_sigma'] if 'second_blur_sigma' in optk.keys(
        ) else [0.2, 1.5]
        self.second_betag_range = optk['second_betag_range'] if 'second_betag_range' in optk.keys(
        ) else [0.5, 4]
        self.second_betap_range = optk['second_betap_range'] if 'second_betap_range' in optk.keys(
        ) else [1, 2]
        self.second_sinc_prob = optk['second_sinc_prob'] if 'second_sinc_prob' in optk.keys(
        ) else 0.1

        # a final sinc filter
        self.final_sinc_prob = optk['final_sinc_prob'] if 'final_sinc_prob' in optk.keys(
        ) else 0.8

        # kernel size ranges from 7 to 21
        self.kernel_range_first = [
            2 * v + 1 for v in range(2, round(self.first_blur_kernel_size / 2) + 1)]
        self.kernel_range_second = [
            2 * v + 1 for v in range(2, round(self.first_blur_kernel_size / 2) + 1)]
        # TODO: kernel range is now hard-coded, should be in the configure file
        # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1

        self.jpeger = DiffJPEG(differentiable=False)
        self.usm_sharpener = USMSharp()
        self.text_draw = TextDraw()

    def random_kernels(self):
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range_first)
        if np.random.uniform() < self.first_sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            first_kernel = circular_lowpass_kernel(
                omega_c, kernel_size, pad_to=False)
        else:
            first_kernel = random_mixed_kernels(
                self.first_kernel_list,
                self.first_kernel_prob,
                kernel_size,
                self.first_blur_sigma,
                self.first_blur_sigma, [-math.pi, math.pi],
                self.first_betag_range,
                self.first_betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        first_kernel = np.pad(
            first_kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range_second)
        if np.random.uniform() < self.second_sinc_prob:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            second_kernel = circular_lowpass_kernel(
                omega_c, kernel_size, pad_to=False)
        else:
            second_kernel = random_mixed_kernels(
                self.second_kernel_list,
                self.second_kernel_prob,
                kernel_size,
                self.second_blur_sigma,
                self.second_blur_sigma, [-math.pi, math.pi],
                self.second_betag_range,
                self.second_betap_range,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        second_kernel = np.pad(
            second_kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- sinc kernel ------------------------------------- #
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range_first)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(
                omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        first_kernel = torch.FloatTensor(first_kernel)
        second_kernel = torch.FloatTensor(second_kernel)

        return first_kernel, second_kernel, sinc_kernel

    @torch.no_grad()
    def realesr_process(self, img_origin, random_draw_text=False, random_draw_text_ratio=0.5, resize_bak=False):
        if self.crop_type == "random":
            img_origin = random_crop_arr(img_origin, self.out_size)
        elif self.crop_type == "center":
            img_origin = center_crop_arr(img_origin, self.out_size)
        elif self.crop_type == "multi_scale":
            img_origin = multi_scale_crop(img_origin, self.out_size)
        # self.crop_type is "none"
        else:
            img_origin = np.array(img_origin)
            assert img_origin.shape[:2] == (self.out_size, self.out_size)

        if random_draw_text:
            if np.random.random() < random_draw_text_ratio:
                try:
                    img_origin = self.text_draw.draw_text(img_origin)
                except:
                    img_origin = img_origin

        # hwc, rgb to bgr, [0, 255] to [0, 1], float32
        img_hq = (img_origin[..., ::-1] / 255.0).astype(np.float32)

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_hq = augment(img_hq, self.use_hflip, self.use_rot)

        # [0, 1], BGR to RGB, HWC to CHW
        img_hq = torch.from_numpy(
            img_hq[..., ::-1].transpose(2, 0, 1).copy()).float().unsqueeze(0)

        first_kernel, second_kernel, sinc_kernel = self.random_kernels()

        img_gt, first_kernel, second_kernel, sinc_kernel = img_hq.clone(
        ), first_kernel.clone(), second_kernel.clone(), sinc_kernel.clone()

        if True:
            img_gt = self.usm_sharpener(img_gt)  # shaper gt

        ori_h, ori_w = img_gt.size()[2:4]
        # scale_final = random.randint(4, 16)
        scale_final = 4

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(img_gt, first_kernel)

        # random resize
        updown_type = random.choices(
            ['up', 'down', 'keep'], self.first_resize_prob)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.first_resize_range[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.first_resize_range[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)

        # noise
        if np.random.uniform() < self.first_add_noise_prob:
            gray_noise_prob = self.first_gray_noise_prob
            if np.random.uniform() < self.first_gaussian_noise_prob:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.first_noise_range, clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.first_poisson_scale_range,
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.first_jpeg_range)
        out = torch.clamp(out, 0, 1)
        out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.second_blur_prob:
            out = filter2D(out, second_kernel)
        # random resize
        updown_type = random.choices(
            ['up', 'down', 'keep'], self.second_resize_prob)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.second_resize_range[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.second_resize_range[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out, size=(int(ori_h / scale_final * scale), int(ori_w / scale_final * scale)), mode=mode)
        # noise
        if np.random.uniform() < self.second_add_noise_prob:
            gray_noise_prob = self.second_gray_noise_prob
            if np.random.uniform() < self.second_gaussian_noise_prob:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.second_noise_range, clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.second_poisson_scale_range,
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(
                ori_h // scale_final, ori_w // scale_final), mode=mode)
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(
                *self.second_jpeg_range)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(
                *self.second_jpeg_range)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(
                ori_h // scale_final, ori_w // scale_final), mode=mode)
            out = filter2D(out, sinc_kernel)

        if np.random.uniform() < self.gray_prob:
            out = rgb_to_grayscale(out, num_output_channels=1)

        if np.random.uniform() < self.color_jitter_prob:
            brightness = self.opt.get('brightness', (0.5, 1.5))
            contrast = self.opt.get('contrast', (0.5, 1.5))
            saturation = self.opt.get('saturation', (0, 1.5))
            hue = self.opt.get('hue', (-0.1, 0.1))
            out = color_jitter_pt(out, brightness, contrast, saturation, hue)

        if resize_bak:
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h, ori_w), mode=mode)

        # clamp and round
        img_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        return img_hq, img_lq

    @torch.no_grad()
    def firstorder_degra_process(self, img_pil, random_draw_text=False, random_draw_text_ratio=0.5, resize_bak=True):
        if self.crop_type == "random":
            img_hq = random_crop_arr(img_pil, self.out_size)
        elif self.crop_type == "center":
            img_hq = center_crop_arr(img_pil, self.out_size)
        elif self.crop_type == "multi_scale":
            img_hq = multi_scale_crop(img_pil, self.out_size)
        else:
            img_hq = np.array(img_pil).astype(np.float32)
            assert img_pil.shape[:2] == (self.out_size, self.out_size)

        if random_draw_text:
            if np.random.random() < random_draw_text_ratio:
                try:
                    img_hq = self.text_draw.draw_text(img_hq)
                except:
                    img_hq = img_hq

        # numpy [0, 255] (h,w,c) ->  torch [0, 1] (1, c, h,w)
        img_origin = torch.from_numpy(
            (np.array(img_hq.copy()) / 255.0).transpose(2, 0, 1).copy()).float().unsqueeze(0)
        first_kernel, _, _ = self.random_kernels()
        img_gt, first_kernel = img_origin.clone(), first_kernel.clone()
        ori_h, ori_w = img_gt.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(img_gt, first_kernel)

        # random resize
        updown_type = random.choices(
            ['up', 'down', 'keep'], self.first_resize_prob)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.first_resize_range[1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.first_resize_range[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)

        # noise
        if np.random.uniform() < self.first_add_noise_prob:
            gray_noise_prob = self.first_gray_noise_prob
            if np.random.uniform() < self.first_gaussian_noise_prob:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.first_noise_range, clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.first_poisson_scale_range,
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

        # JPEG compression
        if np.random.uniform() > 0.5:
            jpeg_p = out.new_zeros(out.size(0)).uniform_(
                *self.first_jpeg_range)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)

            if resize_bak:
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h, ori_w), mode=mode)
        else:
            if resize_bak:
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h, ori_w), mode=mode)

            jpeg_p = out.new_zeros(out.size(0)).uniform_(
                *self.first_jpeg_range)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)

        # torch [0, 1] (1, c, h,w) ->  numpy [0, 255] (h,w,c)
        img_lq = torch.clamp((out * 255.0).round(), 0,
                             255).cpu().numpy()[0, :, :, :].transpose(1, 2, 0)

        return img_hq, img_lq

    @torch.no_grad()
    def nice_deg_pipe(self, img_pil, random_draw_text=False, random_draw_text_ratio=0.2):
        """
        输入图像：cv2读取结果
        """
        # 高斯模糊
        if self.crop_type == "random":
            img_hq = random_crop_arr(img_pil, self.out_size)
        elif self.crop_type == "center":
            img_hq = center_crop_arr(img_pil, self.out_size)
        elif self.crop_type == "multi_scale":
            img_hq = multi_scale_crop(img_pil, self.out_size)
        else:
            img_hq = np.array(img_pil).astype(np.float32)
            assert img_pil.shape[:2] == (self.out_size, self.out_size)

        if random_draw_text:
            if np.random.random() < random_draw_text_ratio:
                try:
                    img_hq = self.text_draw.draw_text(img_hq)
                except:
                    img_hq = img_hq

        gaussianblur_flag = np.random.uniform() < 0.6
        resize_flag = True
        JEPG_flag = np.random.uniform() < 0.6
        if gaussianblur_flag:
            img_lq = GaussianBlur(img_hq.copy())
        if resize_flag:
            img_lq = resize_degradation(img_hq.copy())
        if JEPG_flag:
            img_lq = JEPG_degradation(img_hq.copy())

        return img_hq, img_lq

    @torch.no_grad()
    def deg_pipe(self, source):
        """
        输入图像：cv2读取结果
        """
        # 高斯模糊
        k_size = random.randint(1, 5) * 2 + 1
        sigma = np.random.choice(np.arange(3, 8, step=0.1))
        source = cv2.GaussianBlur(source, (k_size, k_size), sigma)
        # 随机resize
        ori_h, ori_w = source.shape[0], source.shape[1]
        scale_factor = random.uniform(0.5, 1.5)
        resize_mode = random.choice(
            [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
        resized_img = cv2.resize(
            source, None, fx=scale_factor, fy=scale_factor, interpolation=resize_mode)
        # JPEG压缩与resize back
        quality_parameter = random.randint(70, 99)  # JPEG 压缩的质量参数，范围为 0 到 100
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality_parameter]
        if np.random.uniform() < 0.5:
            # 先resize回来，再加JPEG压缩
            img = cv2.resize(resized_img, (ori_w, ori_h),
                             interpolation=resize_mode)
            _, encimg = cv2.imencode('.jpg', img, encode_param)
            return cv2.imdecode(encimg, 1)
        else:
            # 先加JPEG压缩，再resize回来
            _, encimg = cv2.imencode('.jpg', resized_img, encode_param)
            decimg = cv2.imdecode(encimg, 1)
            return cv2.resize(decimg, (ori_w, ori_h), interpolation=resize_mode)

    @torch.no_grad()
    def deg_pipe_1(self, source):
        """
        输入图像：cv2读取结果
        lsdir基准效果
        """
        # 高斯模糊
        k_size = random.randint(0, 10) * 2 + 1
        sigma = np.random.choice(np.arange(1, 15, step=0.1))
        source = cv2.GaussianBlur(source, (k_size, k_size), sigma)
        # 随机resize
        ori_h, ori_w = source.shape[0], source.shape[1]
        # scale_factor = random.uniform(0.5, 1.5)
        scale_factor = random.uniform(0.25, 4)
        resize_mode = random.choice(
            [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
        resized_img = cv2.resize(
            source, None, fx=scale_factor, fy=scale_factor, interpolation=resize_mode)
        # JPEG压缩与resize back
        quality_parameter = random.randint(50, 99)  # JPEG 压缩的质量参数，范围为 0 到 100
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality_parameter]
        if np.random.uniform() < 0.5:
            # 先resize回来，再加JPEG压缩
            img = cv2.resize(resized_img, (ori_w, ori_h),
                             interpolation=resize_mode)
            _, encimg = cv2.imencode('.jpg', img, encode_param)
            return cv2.imdecode(encimg, 1)
        else:
            # 先加JPEG压缩，再resize回来
            _, encimg = cv2.imencode('.jpg', resized_img, encode_param)
            decimg = cv2.imdecode(encimg, 1)
            return cv2.resize(decimg, (ori_w, ori_h), interpolation=resize_mode)
