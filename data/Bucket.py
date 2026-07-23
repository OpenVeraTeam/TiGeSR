import json
import queue
import time
from torch.utils import data as data
from torchvision import transforms
from PIL import Image, ImageFilter
import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tF
import os
import random
import threading
from data.utils import find_files_directory, img2tensor

class AspectRatioBatchSampler(torch.utils.data.Sampler):
    def __init__(self, buckets, batch_size, shuffle=True, drop_last=True, 
                 num_replicas=1, rank=0):
        """
        DDP-compatible batch sampler with preloaded batches per bucket.

        Args:
            buckets (Dict[Tuple[int,int], List[Any]]): Dict mapping bucket_id -> list of samples
            batch_size (int): Batch size per GPU
            shuffle (bool): Shuffle batch order each epoch
            drop_last (bool): Drop last incomplete batch
            num_replicas (int): Number of distributed processes
            rank (int): Rank of the current process
        """
        self.buckets = buckets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_replicas = num_replicas
        self.rank = rank

        self.batch_list = self._build_batches()

    def _build_batches(self):
        batch_list = []
        for bucket_id, samples in self.buckets.items():
            indices = list(range(len(samples)))
            if self.shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                if len(batch_indices) < self.batch_size and self.drop_last:
                    continue
                batch_list.append((bucket_id, batch_indices))
        
        # 分布式处理：将批次分配给不同进程
        num_batches = len(batch_list)
        num_batches_per_replica = math.ceil(num_batches / self.num_replicas)
        
        # 确保每个进程有相同数量的批次（填充）
        total_batches = num_batches_per_replica * self.num_replicas
        
        if len(batch_list) < total_batches:
            batch_list.extend(batch_list[:total_batches - len(batch_list)])
        
        # 为当前进程选择对应的批次
        rank_batches = batch_list[self.rank:total_batches:self.num_replicas]
        return rank_batches

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batch_list)
        for bucket_id, idx in self.batch_list:
            yield {"bucket_size": bucket_id, "idx": idx}

    def __len__(self):
        return len(self.batch_list)


from concurrent.futures import ThreadPoolExecutor, as_completed
class VariableSizeDataLoader:
    def __init__(self, dataset, batch_sampler,num_workers=8, prefetch_factor=2):
        """
        DataLoader to return variable-sized batches.

        Args:
            sampler (AspectRatioBatchSampler): The sampler object.
        """
        self.batch_sampler = batch_sampler
        self.dataset = dataset
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.prefetch_queue = queue.Queue(maxsize=prefetch_factor)
        self.stop_event = threading.Event()  # 添加停止事件

    def __len__(self):
        return len(self.batch_sampler)
    
    def _prefetch_data(self):
        try:
            for batch in self.batch_sampler:
                if self.stop_event.is_set():
                    break

                bucket_size = batch["bucket_size"]
                idxs = batch["idx"]

                results = []
                # 使用with语句确保线程池正确关闭
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    try:
                        futures = [executor.submit(self.dataset.__getitem__, (bucket_size, i)) for i in idxs]
                        for future in as_completed(futures):
                            try:
                                results.append(future.result())
                            except Exception as e:
                                print(f"处理单个项目时出错: {e}")
                                # 可以选择跳过或中断
                    except Exception as e:
                        print(f"提交任务到线程池时出错: {e}")
                        continue
                    
                # 检查是否获取了足够的结果
                if len(results) < len(idxs):
                    print(f"警告: 只获取了{len(results)}/{len(idxs)}个结果")
                    continue

                # ... 处理结果的代码 ...
        except Exception as e:
            print(f"预取线程出现未捕获的异常: {e}")

    def _load_item(self, bucket_size, idx):
        return self.dataset[bucket_size, idx]

    # def __iter__(self):
    #     self.stop_event.clear()  # 重置停止事件
    #     prefetch_thread = threading.Thread(target=self._prefetch_data)
    #     prefetch_thread.daemon = True
    #     prefetch_thread.start()

    #     try:
    #         for _ in range(len(self.batch_sampler)):
    #             batch = self.prefetch_queue.get(timeout=60)
    #             if batch is None:  # 检查是否为错误信号
    #                 break
    #             # ... 处理批次的代码 ...
    #             yield batch
    #     finally:
    #         # 确保在迭代结束时停止预取线程
    #         self.stop_event.set()
    #         # 清空队列，避免预取线程卡在put操作
    #         while not self.prefetch_queue.empty():
    #             try:
    #                 self.prefetch_queue.get_nowait()
    #             except:
    #                 pass

    def __iter__(self):
        for batch in self.batch_sampler:
            bucket_size = batch["bucket_size"]
            idxs = batch["idx"]

            # img_lq, img_hq, img_mask, img_condition,is_ver = [], [], [], [],[]
            # for i in idxs:
            #     item = self.dataset[bucket_size, i]
            #     img_lq.append(item["lq"])
            #     img_hq.append(item["hq"])
            #     img_mask.append(item["mask"])
            #     img_condition.append(item["text"])

             # 使用线程池并行加载数据
            with ThreadPoolExecutor(max_workers=min(self.num_workers, len(idxs))) as executor:
                futures = [executor.submit(self.dataset.__getitem__, (bucket_size, i)) for i in idxs]
                results = [future.result() for future in as_completed(futures)]

            img_lq = [item["lq"] for item in results]
            img_hq = [item["hq"] for item in results]
            img_mask = [item["mask"] for item in results]
            img_condition = [item["text"] for item in results]

            img_lq = torch.stack(img_lq).cuda()
            img_hq = torch.stack(img_hq).cuda()
            img_mask = torch.stack(img_mask).cuda()
            # img_condition = torch.stack(img_condition)  # if it's tensor

            yield {
                "lq": img_lq,
                "hq": img_hq,
                "mask": img_mask,
                "text": img_condition,
            }


# class BucketDataset(data.Dataset):
#     def __init__(self, dataset_folders: str,max_ratio: int = 20, crop_rate=0.3,text_dropout_prob: float = 0.2, 
#                  image_dropout_prob: float = 0.3):
#         super().__init__()
#         if isinstance(dataset_folders, str):
#             dataset_folders = [dataset_folders]

#         self.input_folders = dataset_folders
#         self.crop_rate = crop_rate  # 裁切增广率
#         self.len_buckets = 0
#         self.bucket_contents = {}
#         self.patch_height = 128
#         self.text_dropout_prob = text_dropout_prob  # 文本dropout概率
#         self.image_dropout_prob = image_dropout_prob  # 图像dropout概率
#         bucket_index = np.arange(1,max_ratio)
#         for folder in self.input_folders:
#             for i in bucket_index:
#                 bucket_name = str(i)
#                 img_files = sorted(find_files_directory(
#                     os.path.join(folder, bucket_name, "gt")))
#                 if bucket_name not in self.bucket_contents:
#                     self.bucket_contents[bucket_name] = []  # 初始化空列表
#                 self.len_buckets += len(img_files)
#                 self.bucket_contents[bucket_name].extend(img_files)  # 合并列表

#     def __len__(self) -> int:
#         return self.len_buckets

#     def apply_image_dropout(self, image):
#         """对图像应用随机dropout"""
#         if random.random() < self.image_dropout_prob:
#             # 创建随机噪声掩码
#             mask = torch.rand_like(image) > 0.15
#             # 应用掩码，将随机像素置为0
#             return image * mask
#         return image
    
#     def apply_text_dropout(self, text):
#         """对文本应用部分随机dropout"""
#         if not text or random.random() >= self.text_dropout_prob:
#             return text
            
#         # 方法1：按字符级别进行dropout
#         if random.random() < 0.5:
#             # 将文本转换为字符列表
#             chars = list(text)
#             # 随机决定保留哪些字符
#             keep_mask = [random.random() > self.text_dropout_prob for _ in range(len(chars))]
#             # 至少保留一个字符
#             if not any(keep_mask) and chars:
#                 keep_mask[random.randint(0, len(chars) - 1)] = True
#             # 应用掩码
#             result = ''.join([c for i, c in enumerate(chars) if keep_mask[i]])
#             return result
#         # 方法2：按单词级别进行dropout
#         else:
#             # 将文本分割为单词
#             words = text.split()
#             if not words:
#                 return text
#             # 随机决定保留哪些单词
#             keep_mask = [random.random() > self.text_dropout_prob for _ in range(len(words))]
#             # 至少保留一个单词
#             if not any(keep_mask) and words:
#                 keep_mask[random.randint(0, len(words) - 1)] = True
#             # 应用掩码
#             result = ' '.join([w for i, w in enumerate(words) if keep_mask[i]])
#             return result

#     def __getitem__(self, tuple_idx):
#         bucket_id, idx = tuple_idx
#         image_hq_path = self.bucket_contents[bucket_id][idx]
#         image_lq_path = image_hq_path.replace("/gt/",'/lr_randomsff/')
#         image_mask_path = image_hq_path.replace("/gt/",'/mask/')
#         patch_width = self.patch_height * int(bucket_id)

#         img_pil_hq = Image.open(image_hq_path).convert("RGB")
#         img_pil_lq = Image.open(image_lq_path).convert("RGB")
#         img_mask_pil = Image.open(image_mask_path).convert("RGB")
#         W, H = img_pil_hq.size

#         text = os.path.splitext(os.path.basename(image_hq_path))[0]

#         # 处理旋转
#         if H/W >= 2:
#             img_pil_hq = img_pil_hq.transpose(Image.ROTATE_90)
#             img_pil_lq = img_pil_lq.transpose(Image.ROTATE_90)
#             img_mask_pil = img_mask_pil.transpose(Image.ROTATE_90)
#             W, H = img_pil_hq.size  # 更新旋转后的尺寸

#         top_crop = 0
#         bottom_crop = 0
#         # 随机裁切高度增广
#         if random.random() < self.crop_rate:  # 30%的概率应用高度裁切增广
#             # 随机确定裁切的总比例(0%~15%)
#             crop_ratio = random.uniform(0, 0.3)
#             crop_height = int(H * crop_ratio)

#             # 随机分配裁切到顶部和底部
#             top_crop = random.randint(0, crop_height)
#             bottom_crop = crop_height - top_crop

#             # 应用裁切
#             if crop_height > 0:
#                 new_h = H - crop_height
#                 img_pil_hq = img_pil_hq.crop((0, top_crop, W, H - bottom_crop))
#                 img_pil_lq = img_pil_lq.crop((0, top_crop, W, H - bottom_crop))
#                 img_mask_pil = img_mask_pil.crop((0, top_crop, W, H - bottom_crop))
#         left_crop = 0
#         right_crop = 0
#         # 随机裁切宽度增广
#         if random.random() < self.crop_rate:  # 30%的概率应用宽度裁切增广
#             # 计算最大允许裁切宽度，确保不超过 W/len(text)*0.3
#             text_length = max(1, len(text))  # 避免除零错误
#             max_crop_width = int(W / text_length * 0.3)

#             # 随机确定裁切的总宽度(0到最大允许值)
#             crop_width = random.randint(0, max_crop_width)

#             # 随机分配裁切到左侧和右侧
#             left_crop = random.randint(0, crop_width)
#             right_crop = crop_width - left_crop

#             # 应用裁切
#             if crop_width > 0:
#                 new_w = W - crop_width
#                 img_pil_hq = img_pil_hq.crop((left_crop, 0, W - right_crop, H))
#                 img_pil_lq = img_pil_lq.crop((left_crop, 0, W - right_crop, H))
#                 img_mask_pil = img_mask_pil.crop((left_crop, 0, W - right_crop, H))

#         # 调整大小
#         img_pil_hq = img_pil_hq.resize((patch_width, self.patch_height), Image.LANCZOS)
#         img_pil_lq = img_pil_lq.resize((patch_width, self.patch_height), Image.LANCZOS)
#         img_mask_pil = img_mask_pil.resize((patch_width, self.patch_height), Image.LANCZOS)

#         img_hq = img2tensor([np.array(img_pil_hq) / 255.0],
#                             bgr2rgb=False, float32=True)[0]
#         img_lq = img2tensor([np.array(img_pil_lq) / 255.0],
#                             bgr2rgb=False, float32=True)[0]
#         img_mask = img2tensor([np.array(img_mask_pil) / 255.0],
#                             bgr2rgb=False, float32=True)[0]

#         example = dict()
#         example['hq'] = img_hq
#         example['lq'] = img_lq
#         example['mask'] = img_mask
#         example["text"] = text
#         return example
    

class BucketDataset(data.Dataset):
    def __init__(self, dataset_folders: str,json_anno=False):
        super().__init__()
        self.crop_rate = 0
        self.json_anno = json_anno
        if isinstance(dataset_folders, str):
            dataset_folders = [dataset_folders]
        self.input_folders = dataset_folders
        self.len_buckets = 0
        self.bucket_contents = {}
        bucket_index = np.arange(-15,30)
        # img_files = []
        self.text_anno = {}
        for folder in self.input_folders:
            if "LSDIR" in folder or "RealCE_mod" in folder or "UltraLens" in folder:
                with open(os.path.join(folder, "anno.json"), 'r') as f:
                    self.text_anno.update(json.load(f))
            for i in bucket_index:
                if i > 0:
                    bucket_name = str(i)
                    try:
                        img_files = sorted(find_files_directory(
                            os.path.join(folder, "horizontal", bucket_name, "gt")))
                        self.len_buckets += len(img_files)
                        if bucket_name not in self.bucket_contents:
                            self.bucket_contents[bucket_name] = []  # 初始化空列表
                        self.bucket_contents[bucket_name].extend(img_files)  # 合并列表
                    except:
                        pass
                elif i < 0 and "en" not in folder:
                    bucket_name = str(i)
                    try:
                        img_files = sorted(find_files_directory(
                            os.path.join(folder, "vertical", bucket_name[1:], "gt")))
                        self.len_buckets += len(img_files)
                        if bucket_name not in self.bucket_contents:
                            self.bucket_contents[bucket_name] = []  # 初始化空列表
                        self.bucket_contents[bucket_name].extend(img_files)  # 合并列表
                    except:
                        pass

    def __len__(self) -> int:
        return self.len_buckets

    def __getitem__(self, tuple_idx):
        bucket_id, idx = tuple_idx
        image_hq_path = self.bucket_contents[bucket_id][idx]
        # 使用字典映射替代多次替换
        paths = {
            'hq': image_hq_path,
            # 'lq': image_hq_path.replace("/gt/", '/lr/') if "LSDIR" in image_hq_path else image_hq_path.replace("/gt/", '/lr_randomsff/'),
            'lq': image_hq_path.replace("/gt/", '/lr/') if "LSDIR" in image_hq_path or "RealCE_mod/bucket" in image_hq_path or "UltraLens" in image_hq_path else image_hq_path.replace("/gt/", '/lr_resized/'),
            'mask': image_hq_path.replace("/gt/", '/mask/')
        }

        # 一次性打开所有图片
        images_pil = {k: Image.open(path).convert("RGB") for k, path in paths.items()}

        W, H = images_pil['hq'].size
        if "LSDIR" in image_hq_path or "RealCE_mod" in image_hq_path:
            # 如果是LSDIR数据集，直接使用文件名作为文本
            text = self.text_anno.get(os.path.splitext(os.path.basename(image_hq_path))[0], "")
        else:
            text = os.path.splitext(os.path.basename(image_hq_path))[0]


        # 计算patch尺寸
        top_crop = 0
        bottom_crop = 0
        left_crop = 0
        right_crop = 0
        if int(bucket_id) > 0: # 正常水平图
            patch_height = 128
            patch_width = patch_height * int(bucket_id)
        #     if random.random() < self.crop_rate:  # 30%的概率应用宽度裁切增广
        #         crop_ratio = random.uniform(0, 0.25)
        #         crop_height = int(H * crop_ratio)

        #         # 随机分配裁切到顶部和底部
        #         top_crop = random.randint(0, crop_height)
        #         bottom_crop = crop_height - top_crop

        #     if random.random() < self.crop_rate:  # 30%的概率应用宽度裁切增广
        #         # 计算最大允许裁切宽度，确保不超过 W/len(text)*0.3
        #         text_length = max(1, len(text))  # 避免除零错误
        #         max_crop_width = int(W / text_length * 0.25)

        #         # 随机确定裁切的总宽度(0到最大允许值)
        #         crop_width = random.randint(0, max_crop_width)

        #         # 随机分配裁切到左侧和右侧
        #         left_crop = random.randint(0, crop_width)
        #         right_crop = crop_width - left_crop
        else: # 竖直图
            patch_width = 128
            patch_height = patch_width * abs(int(bucket_id))
        #     if random.random() < self.crop_rate:  # 30%的概率应用宽度裁切增广
        #         crop_ratio = random.uniform(0, 0.25)
        #         crop_width = int(W * crop_ratio)
        #         left_crop = random.randint(0, crop_width)
        #         right_crop = crop_width - left_crop
            
        #     if random.random() < self.crop_rate:  # 30%的概率应用宽度裁切增广
        #         # 计算最大允许裁切宽度，确保不超过 W/len(text)*0.3
        #         text_length = max(1, len(text))  # 避免除零错误
        #         max_crop_height = int(H / text_length * 0.25)

        #         # 随机确定裁切的总宽度(0到最大允许值)
        #         crop_height = random.randint(0, max_crop_height)
        #         # 随机分配裁切到左侧和右侧
        #         top_crop = random.randint(0, crop_width)
        #         bottom_crop = crop_width - top_crop
        # img_pil_hq = img_pil_hq.crop((left_crop, top_crop, W - right_crop, H-bottom_crop))
        # img_pil_lq = img_pil_lq.crop((left_crop, top_crop, W - right_crop, H-bottom_crop))
        # img_mask_pil = img_mask_pil.crop((left_crop, top_crop, W - right_crop, H-bottom_crop))
        # 调整所有图片大小
        for k in images_pil:
            # images_pil[k] = images_pil[k].crop((left_crop, top_crop, W - right_crop, H-bottom_crop))
            images_pil[k] = images_pil[k].resize((patch_width, patch_height), Image.LANCZOS)

        # 批量转换为tensor
        images_tensor = {k: img2tensor([np.array(img) / 255.0], bgr2rgb=False, float32=True)[0]
                         for k, img in images_pil.items()}

        # 构建返回字典
        example = {
            'hq': images_tensor['hq'],
            'lq': images_tensor['lq'],
            'mask': images_tensor['mask'],
            'text': text,
        }
        return example

from data.degradation.real_esrgan_bsrgan_degradation import real_esrgan_degradation

class BucketDataset_online_deg(BucketDataset):

    def __getitem__(self, tuple_idx):
        example = super().__getitem__(tuple_idx)
        example['lq'] = self.apply_image_dropout(example['lq'])
        example['hq'] = self.apply_image_dropout(example['hq'])
        example['text'] = self.apply_text_dropout(example['text'])
        return example

    def __getitem__(self, tuple_idx):
        bucket_id, idx = tuple_idx
        image_hq_path = self.bucket_contents[bucket_id][idx]
        # image_lq_path = image_hq_path.replace("/gt/",'/lr/')
        image_mask_path = image_hq_path.replace("/gt/",'/mask/')
        patch_width = self.patch_height * int(bucket_id)

        img_pil_hq = Image.open(image_hq_path).convert("RGB")
        # img_pil_lq = Image.open(image_lq_path).convert("RGB")
        img_mask_pil = Image.open(image_mask_path).convert("RGB")
        W, H = img_pil_hq.size

        text = os.path.splitext(os.path.basename(image_hq_path))[0]

        # 处理旋转
        if H/W >= 2:
            img_pil_hq = img_pil_hq.transpose(Image.ROTATE_90)
            # img_pil_lq = img_pil_lq.transpose(Image.ROTATE_90)
            img_mask_pil = img_mask_pil.transpose(Image.ROTATE_90)
            W, H = img_pil_hq.size  # 更新旋转后的尺寸

        # 随机裁切高度增广
        if random.random() < self.crop_rate:  # 30%的概率应用高度裁切增广
            # 随机确定裁切的总比例(0%~15%)
            crop_ratio = random.uniform(0, 0.3)
            crop_height = int(H * crop_ratio)

            # 随机分配裁切到顶部和底部
            top_crop = random.randint(0, crop_height)
            bottom_crop = crop_height - top_crop

            # 应用裁切
            if crop_height > 0:
                new_h = H - crop_height
                img_pil_hq = img_pil_hq.crop((0, top_crop, W, H - bottom_crop))
                # img_pil_lq = img_pil_lq.crop((0, top_crop, W, H - bottom_crop))
                img_mask_pil = img_mask_pil.crop((0, top_crop, W, H - bottom_crop))

        # 随机裁切宽度增广
        if random.random() < self.crop_rate:  # 30%的概率应用宽度裁切增广
            # 计算最大允许裁切宽度，确保不超过 W/len(text)*0.3
            text_length = max(1, len(text))  # 避免除零错误
            max_crop_width = int(W / text_length * 0.3)

            # 随机确定裁切的总宽度(0到最大允许值)
            crop_width = random.randint(0, max_crop_width)

            # 随机分配裁切到左侧和右侧
            left_crop = random.randint(0, crop_width)
            right_crop = crop_width - left_crop

            # 应用裁切
            if crop_width > 0:
                new_w = W - crop_width
                img_pil_hq = img_pil_hq.crop((left_crop, 0, W - right_crop, H))
                # img_pil_lq = img_pil_lq.crop((left_crop, 0, W - right_crop, H))
                img_mask_pil = img_mask_pil.crop((left_crop, 0, W - right_crop, H))

        # 调整大小
        img_pil_hq = img_pil_hq.resize((patch_width, self.patch_height), Image.LANCZOS)
        # img_pil_lq = img_pil_lq.resize((patch_width, self.patch_height), Image.LANCZOS)
        img_mask_pil = img_mask_pil.resize((patch_width, self.patch_height), Image.LANCZOS)

        img_hq = img2tensor([np.array(img_pil_hq) / 255.0],
                            bgr2rgb=False, float32=True)[0]
        # img_lq = img2tensor([np.array(img_pil_lq) / 255.0],
        #                     bgr2rgb=False, float32=True)[0]
        img_mask = img2tensor([np.array(img_mask_pil) / 255.0],
                            bgr2rgb=False, float32=True)[0]
        img_lq = F.interpolate(
            real_esrgan_degradation(img_hq, insf=4),
            (self.patch_height,patch_width),
            mode="nearest"
        )[0]

        example = dict()
        example['hq'] = img_hq
        example['lq'] = img_lq
        example['mask'] = img_mask
        example["text"] = text
        example["vertical"] = H/W >= 2  # 是否竖屏
        return example