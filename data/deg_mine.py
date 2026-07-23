import argparse
import glob
import json
import os
import random
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.utils import save_image



class RealESRGANDataset(data.Dataset):
    def __init__(self, data_path,opt):
        super(RealESRGANDataset, self).__init__()
        self.opt = opt
        if 'crop_size' in opt:
            self.crop_size = opt['crop_size']
        else:
            self.crop_size = 512

        # support multiple type of data: file path and meta data, remove support of lmdb
        self.gt_data = SyntheticDataset(
            image_dataset=data_path,
            output_size=(self.crop_size, self.crop_size),
            min_scale=opt['min_scale'],
            get_pil_image=False
        )

        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(10, 11)]  # kernel size ranges from 7 to 21

        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def __getitem__(self, index):
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi / 2)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi / 4)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi / 2)
            else:
                omega_c = np.random.uniform(np.pi / 4, np.pi / 3)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        data_item = self.gt_data[index]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {
            'name':data_item['name'],
            'gt': data_item['image'],
            'kernel1': kernel,
            'kernel2': kernel2,
            'sinc_kernel': sinc_kernel,
            'size': data_item['size'],
            'sf': data_item['sf']
        }
        return return_d

    def __len__(self):
        return len(self.gt_data)


def realesrgan_degradation(batch, args_degradation, use_usm=True, sf=4,jpeger=DiffJPEG(differentiable=False).cuda(),usm_sharpener = USMSharp().cuda()):
    im_gt = batch['gt'].cuda()
    if use_usm:
        im_gt = usm_sharpener(im_gt)
    im_gt = im_gt.to(memory_format=torch.contiguous_format).float()
    kernel1 = batch['kernel1'].cuda()
    kernel2 = batch['kernel2'].cuda()
    sinc_kernel = batch['sinc_kernel'].cuda()

    ori_h, ori_w = im_gt.size()[2:4]

    # ----------------------- The first degradation process ----------------------- #
    # blur
    out = filter2D(im_gt, kernel1)
    # add noise
    gray_noise_prob = args_degradation['gray_noise_prob']
    if random.random() < args_degradation['gaussian_noise_prob']:
        out = random_add_gaussian_noise_pt(
            out,
            sigma_range=args_degradation['noise_range'],
            clip=True,
            rounds=False,
            gray_prob=gray_noise_prob,
        )
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=args_degradation['poisson_scale_range'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)
    # JPEG compression
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*args_degradation['jpeg_range'])
    out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
    out = jpeger(out, quality=jpeg_p)

    # ----------------------- The second degradation process ----------------------- #
    # blur
    if random.random() < args_degradation['second_blur_prob']:
        out = filter2D(out, kernel2)

    # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
    if random.random() < 0.5:
        # resize back + the final sinc filter
        mode = random.choice(['bilinear', 'bicubic'])
        out = F.interpolate(
            out,
            size=(ori_h // sf,
                  ori_w // sf),
            mode=mode,
            antialias=True
        )
        # 在调用 filter2D 前检查图像大小
        if min(out.shape[2:]) < sinc_kernel.shape[-1]:
            # 如果图像太小，可以跳过 sinc 滤波或调整 kernel
            sinc_kernel_small = F.interpolate(
                sinc_kernel.unsqueeze(0),
                size=(min(out.shape[2:]) // 2 * 2 - 1, min(out.shape[2:]) // 2 * 2 - 1),
                mode='bilinear'
            ).squeeze(0).squeeze(0)
            out = filter2D(out, sinc_kernel_small)
        else:
            out = filter2D(out, sinc_kernel)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*args_degradation['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
    else:
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*args_degradation['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = jpeger(out, quality=jpeg_p)
        # resize back + the final sinc filter
        mode = random.choice(['bilinear', 'bicubic'])
        out = F.interpolate(
            out,
            size=(ori_h // sf,
                  ori_w // sf),
            mode=mode,
            antialias=True
        )
        if min(out.shape[2:]) < sinc_kernel.shape[-1]:
            # 如果图像太小，可以跳过 sinc 滤波或调整 kernel
            sinc_kernel_small = F.interpolate(
                sinc_kernel.unsqueeze(0),
                size=(min(out.shape[2:]) // 2 * 2 - 1, min(out.shape[2:]) // 2 * 2 - 1),
                mode='bilinear'
            ).squeeze(0).squeeze(0)
            out = filter2D(out, sinc_kernel_small)
        else:
            out = filter2D(out, sinc_kernel)

    # clamp and round
    im_lq = torch.clamp(out, 0, 1.0)

    lq, gt = im_lq, im_gt

    gt = torch.clamp(gt, 0, 1)
    lq = torch.clamp(lq, 0, 1)

    return lq, gt


def get_dataset_settings():
    args_training_dataset = {}
    args_training_dataset['min_scale'] = 0.2

    args_training_dataset['queue_size'] = 160
    args_training_dataset['crop_size'] = 512

    args_training_dataset['blur_kernel_size'] = 13
    args_training_dataset['kernel_list'] = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso',
                                            'plateau_aniso']
    args_training_dataset['kernel_prob'] = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    args_training_dataset['sinc_prob'] = 1 # 0.1
    args_training_dataset['blur_sigma'] = [0.4, 3]
    args_training_dataset['betag_range'] = [0.5, 4]
    args_training_dataset['betap_range'] = [1, 2]

    args_training_dataset['blur_kernel_size2'] = 21
    args_training_dataset['kernel_list2'] = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso',
                                             'plateau_aniso']
    args_training_dataset['kernel_prob2'] = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    args_training_dataset['sinc_prob2'] = 0.8 # 0.1
    args_training_dataset['blur_sigma2'] = [0.4, 1.5]
    args_training_dataset['betag_range2'] = [0.5, 4]
    args_training_dataset['betap_range2'] = [1, 2]

    args_training_dataset['final_sinc_prob'] = 0.8 # 0.4
    return args_training_dataset


def get_degradation_settings():
    args_degradation = {}
    # the first degradation process
    args_degradation['gaussian_noise_prob'] = 0.5
    args_degradation['noise_range'] = [1, 30]
    args_degradation['poisson_scale_range'] = [0.5, 1.25]# [0.05, 1.25]
    args_degradation['gray_noise_prob'] = 0.4
    args_degradation['jpeg_range'] = [50, 95] # [40, 95]

    # the second degradation process
    args_degradation['second_blur_prob'] = 1 # 0.8
    args_degradation['jpeg_range2'] = [70, 95]

    args_degradation['gt_size'] = 512
    return args_degradation


def load_image_paths(dataset_path):
    if isinstance(dataset_path, str):
        if os.path.isdir(dataset_path):
            return glob.glob(os.path.join(dataset_path, '*'))
        elif os.path.isfile(dataset_path):
            with open(dataset_path, 'r') as f:
                return [line.strip() for line in f.readlines()]
        else:
            raise ValueError(f"Invalid path: {dataset_path}")
    else:
        return dataset_path


class SyntheticDataset(Dataset):
    def __init__(self, image_dataset, output_size=(512, 512), min_scale=0.2,
                 get_pil_image=False):
        self.image_dataset = load_image_paths(image_dataset)
        self.output_size = output_size
        self.min_scale = min_scale
        self.transform = transforms.ToTensor()
        self.get_pil_image = get_pil_image

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        # Load a large image from the dataset
        image = self.image_dataset[idx]
        name = image.split("/")[-1]
        # print(image)
        if isinstance(image, str):
            if os.path.isdir(image):
                raise ValueError(f"Expected file path but got directory: {image}")
            image = Image.open(image).convert("RGB")
            
        elif isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image).convert("RGB")
        W,H = image.size
        sf = math.ceil(32 / min(W, H))
        image = image.resize((round(W*sf),round(H*sf)), Image.BICUBIC)
        return {
            "name": name,
            "image": image if self.get_pil_image else self.transform(image),
            "size": (W, H),
            "sf": sf
        }


def custom_collate_fn(batch):
    batch_dict = {}
    for key in batch[0].keys():
        if key == 'anno':
            # Keep 'anno' as a list without stacking to avoid size mismatch error
            batch_dict[key] = [item[key] for item in batch]
        elif key == 'name':
            # Keep 'name' as a string without stacking to avoid size mismatch error
            batch_dict[key] = [item[key] for item in batch]
        else:
            # Stack other keys (e.g., 'data') as usual
            batch_dict[key] = torch.stack([item[key] for item in batch])

    return batch_dict


def biased_random_choice():
    # 使用正态分布，均值为8，标准差适当调整
    # 然后对结果进行四舍五入并限制在[4,15]范围内
    mean = 12
    std_dev = 1  # 调整标准差使12有适当的概率
    
    while True:
        value = int(round(np.random.normal(mean, std_dev)))
        if 4 <= value <= 15:
            return value


if __name__ == '__main__':
    # root_path = "/data/vjuicefs_ai_camera_llm/intern_data/minxing/datasets/CTR"
    parser = argparse.ArgumentParser(description='处理路径参数')
    parser.add_argument('--root_path', type=str, 
                        default="/data/vjuicefs_ai_camera_llm/intern_data/minxing/datasets/rendered/81/train/horizontal",
                        help='数据根路径')
    
    args = parser.parse_args()
    root_path = args.root_path
    gt_path = os.path.join(root_path, 'gt')
    # lr_path = os.path.join(root_path, 'lr_gt8')
    sr_bicubic_path = os.path.join(root_path, 'lr_normal_blur_kernel')
    os.makedirs(gt_path, exist_ok=True)
    # os.makedirs(lr_path, exist_ok=True)
    os.makedirs(sr_bicubic_path, exist_ok=True)

    epochs = 1
    batch_size = 1
    dataset_settings = get_dataset_settings()
    train_dataset = RealESRGANDataset(gt_path,dataset_settings)
    train_dataloader = data.DataLoader(train_dataset, shuffle=False, batch_size=batch_size,
                                       num_workers=16, drop_last=False, collate_fn=custom_collate_fn)
    degradation_settings = get_degradation_settings()
    step = 0
    bar = tqdm(range(len(train_dataset) * epochs))
    jpeger = DiffJPEG(differentiable=False).cuda()
    usm_sharpener = USMSharp().cuda()  # do usm sharpening
    # json_dict = []
    with torch.no_grad():
        for epoch in range(epochs):
            for num_batch, train_batch in enumerate(train_dataloader):
                # sf = biased_random_choice()
                # sf = 8
                B,C,W,H = train_batch['gt'].shape
                # sf = math.ceil(min(W, H) / 24)
                sf = 4
                ori_size = train_batch['size']
                lr_batch, gt_batch = realesrgan_degradation(train_batch, args_degradation=degradation_settings,sf = sf,jpeger=jpeger,usm_sharpener=usm_sharpener)
                sr_bicubic_batch = F.interpolate(lr_batch, size=(ori_size[1],ori_size[0]), mode='bicubic')

                for i in range(batch_size):
                    lr = lr_batch[i, ...]
                    gt = gt_batch[i, ...]
                    name = train_batch['name'][i]
                    sr_bicubic = sr_bicubic_batch[i, ...]
                    json_dict= {
                        'name':train_batch['name'],
                        'kernel1': train_batch["kernel1"].cpu().numpy().tolist(),
                        'kernel2': train_batch["kernel2"].cpu().numpy().tolist(),
                        'sinc_kernel': train_batch["sinc_kernel"].cpu().numpy().tolist()
                    }
                    step += 1
                    bar.update()
                    bar.set_description('process {} images...'.format(step))

                    # lr_save_path = os.path.join(lr_path, '{}.png'.format(str(step).zfill(7)))
                    # gt_save_path = os.path.join(gt_path, '{}.png'.format(str(step).zfill(7)))
                    # sr_bicubic_save_path = os.path.join(sr_bicubic_path, '{}.png'.format(str(step).zfill(7)))
                    # lr_save_path = os.path.join(lr_path, name)
                    # sr_bicubic_save_path = os.path.join(sr_bicubic_path, str(sf)+ "_" + name)
                    sr_bicubic_save_path = os.path.join(sr_bicubic_path, name)

                    # saver.save_image(lr, save_path=lr_save_path)
                    save_image(sr_bicubic, sr_bicubic_save_path)
                    # json_save_path = os.path.join(sr_bicubic_path, str(sf) + "_" + name.replace('.png', '.json'))
                    # with open(json_save_path, 'w', encoding='utf-8') as f:
                    #     json.dump(json_dict, f, indent=4, ensure_ascii=False)  # 只保存最新添加的字典

                del lr_batch, gt_batch, sr_bicubic_batch
                torch.cuda.empty_cache()
