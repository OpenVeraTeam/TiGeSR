'''
Copyright (c) Alibaba, Inc. and its affiliates.
'''
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
import math
import traceback
from easydict import EasyDict as edict
import time
from models.ocr_recog.RecModel import RecModel
import torch
import torch.nn.functional as F
from skimage.transform._geometric import _umeyama as get_sym_mat
import torch.nn as nn


def min_bounding_rect(img):
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print('Bad contours, using fake bbox...')
        return np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    max_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    # sort
    x_sorted = sorted(box, key=lambda x: x[0])
    left = x_sorted[:2]
    right = x_sorted[2:]
    left = sorted(left, key=lambda x: x[1])
    (tl, bl) = left
    right = sorted(right, key=lambda x: x[1])
    (tr, br) = right
    if tl[1] > bl[1]:
        (tl, bl) = (bl, tl)
    if tr[1] > br[1]:
        (tr, br) = (br, tr)
    return np.array([tl, tr, br, bl])


def adjust_image(box, img):
    pts1 = np.float32([box[0], box[1], box[2], box[3]])
    width = max(np.linalg.norm(pts1[0]-pts1[1]), np.linalg.norm(pts1[2]-pts1[3]))
    height = max(np.linalg.norm(pts1[0]-pts1[3]), np.linalg.norm(pts1[1]-pts1[2]))
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    # get transform matrix
    M = get_sym_mat(pts1, pts2, estimate_scale=True)
    C, H, W = img.shape
    T = np.array([[2 / W, 0, -1], [0, 2 / H, -1], [0, 0, 1]])
    theta = np.linalg.inv(T @ M @ np.linalg.inv(T))
    theta = torch.from_numpy(theta[:2, :]).unsqueeze(0).type(torch.float32).to(img.device)
    grid = F.affine_grid(theta, torch.Size([1, C, H, W]), align_corners=True)
    result = F.grid_sample(img.unsqueeze(0), grid, align_corners=True)
    result = torch.clamp(result.squeeze(0), 0, 255)
    # crop
    result = result[:, :int(height), :int(width)]
    return result

def adjust_image_cv2(box, img):
    # img: (C,H,W), box: 4x2 in img坐标（顺序 tl,tr,br,bl 已确认）
    C, H, W = img.shape
    pts1 = np.float32([box[0], box[1], box[2], box[3]])
    # 估计输出矩形的目标尺寸（像素个数，整数）
    edge_top    = np.linalg.norm(pts1[0] - pts1[1])
    edge_bottom = np.linalg.norm(pts1[3] - pts1[2])
    edge_left   = np.linalg.norm(pts1[0] - pts1[3])
    edge_right  = np.linalg.norm(pts1[1] - pts1[2])
    
    out_W = int(round(max(edge_top, edge_bottom)))
    out_H = int(round(max(edge_left, edge_right)))
    
    # 避免0
    out_W = max(out_W, 1)
    out_H = max(out_H, 1)
    
    # 像素中心模型：目标矩形的四点坐标用 [0,0],[out_W-1,0],[out_W-1,out_H-1],[0,out_H-1]
    dst = np.float32([[0, 0],
                      [out_W - 1, 0],
                      [out_W - 1, out_H - 1],
                      [0, out_H - 1]])
    
    # 用OpenCV算单应（更稳），得到 src(img)->dst(rect) 的 M
    M = cv2.getPerspectiveTransform(pts1.astype(np.float32), dst.astype(np.float32))
    
    # 直接透视裁剪到 out_H x out_W
    # 注意 dsize 顺序 (width, height)
    np_img = img.permute(1,2,0).contiguous().detach().cpu().numpy()
    cropped = cv2.warpPerspective(np_img, M, (out_W, out_H),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    out = torch.from_numpy(cropped).permute(2,0,1).to(img.device, dtype=img.dtype)
    return out

def restore_image(box, img, ori_H,ori_W):
    pts2 = np.float32([box[0], box[1], box[2], box[3]])
    C, H, W = img.shape
    print(H,W)
    if W > ori_W:
        img = F.interpolate(
            img.unsqueeze(0), (round(H / W * ori_W), ori_W), mode="nearest"
        ).squeeze(0)
        # print(img.shape)
        C, H, W = img.shape
    if H > ori_H:
        img = F.interpolate(img.unsqueeze(0),(ori_H,round(W/H*ori_H))).squeeze(0)
        # print(img.shape)
        C, H, W = img.shape
    C, H, W = img.shape
    # print(img.shape)
    pts1 = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    # get transform matrix
    M = get_sym_mat(pts1, pts2, estimate_scale=False)
    T = np.array([[2 / ori_W, 0, -1], [0, 2 / ori_H, -1], [0, 0, 1]])
    theta = np.linalg.inv(T @ M @ np.linalg.inv(T))
    theta = torch.from_numpy(theta[:2, :]).unsqueeze(0).type(torch.float32).to(img.device)
    padded_img = torch.zeros((1, C, ori_H, ori_W), device=img.device)
    padded_img[:,:C,:H,:W] = img
    grid = F.affine_grid(theta, torch.Size([1, C, ori_H, ori_W]), align_corners=True)
    result = F.grid_sample(padded_img, grid, align_corners=True)
    result = torch.clamp(result.squeeze(0), 0, 255)
    # # crop
    # result = result[:, :int(height), :int(width)]
    return result[:,:ori_H,:ori_W]


def T_mat(w, h):
    # align_corners=True 对应的像素中心归一化
    return np.array([[2.0 / (w - 1), 0.0, -1.0],
                     [0.0, 2.0 / (h - 1), -1.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)

def debug_homography(box, img_shape, ori_H, ori_W, use_cv=False):
    C,sub_H,sub_W = img_shape
    pts1 = np.float32([[0, 0], [sub_W - 1, 0], [sub_W - 1, sub_H - 1], [0, sub_H - 1]])
    pts2 = np.float32(box).astype(np.float32)

    if use_cv:
        import cv2
        M = cv2.getPerspectiveTransform(pts1, pts2)  # 明确方向 src->dst
    else:
        M = get_sym_mat(pts1, pts2, estimate_scale=False)  # 你自己的

    T_in  = T_mat(sub_W, sub_H)
    T_out = T_mat(ori_W,  ori_H)

    Minv = np.linalg.inv(M)
    A = T_in @ Minv @ np.linalg.inv(T_out)

    # 把输出四点（像素中心）映回输入子图像素坐标
    ones = np.ones((4, 1), dtype=np.float32)
    pts2_h = np.concatenate([pts2, ones], axis=1).T  # 3x4
    pts2_norm = (T_out @ pts2_h)                    # 3x4
    pts_in_norm = (A @ pts2_norm)                   # 3x4
    pts_in = (pts_in_norm[:2, :] / pts_in_norm[2:, :]).T  # 4x2

    print("Expect near [[0,0],[W-1,0],[W-1,H-1],[0,H-1]] in same order:")
    print(np.round(pts_in, 3))
    return pts_in, M, A


def restore_image1(box, img, ori_H, ori_W):
    # img: (C, H, W) 子图
    # box: 4x2，目标画布上的四点，顺序已确认正确（例如 tl,tr,br,bl）
    C, H, W = img.shape
    print("not use cv")
    pts_in, M, A = debug_homography(box,img.shape, ori_H, ori_W)
    print("use_cv")
    pts_in, M, A = debug_homography(box,img.shape, ori_H, ori_W,use_cv=True)
    # 1) 源点：像素中心
    pts1 = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    pts2 = np.float32([box[0], box[1], box[2], box[3]])  # 目标画布上的像素中心坐标

    # 2) 求从源(输入子图)到目标(输出画布)的 3x3 单应矩阵
    M = get_sym_mat(pts1, pts2, estimate_scale=False)  # 确认方向是 src->dst

    # 3) 与 align_corners=True 一致的归一化矩阵（像素中心）
    def T_mat(w, h):
        return np.array([[2.0 / (w - 1), 0.0, -1.0],
                         [0.0, 2.0 / (h - 1), -1.0],
                         [0.0, 0.0, 1.0]], dtype=np.float32)

    T_in = T_mat(W, H)             # 输入子图的归一化（到 [-1,1]）
    T_out = T_mat(ori_W, ori_H)    # 输出画布的归一化（到 [-1,1]）

    # 4) 采样从输出网格到输入图的映射：x_in_norm = T_in * inv(M) * inv(T_out) * x_out_norm
    Minv = np.linalg.inv(M)
    A = T_in @ Minv @ np.linalg.inv(T_out)
    theta = torch.from_numpy(A[:2, :]).unsqueeze(0).to(img.device, dtype=torch.float32)

    # 5) 直接从原始 img 采样到目标画布，不对 img 做额外 resize
    img_batched = img.unsqueeze(0)  # (1, C, H, W)
    grid = F.affine_grid(theta, torch.Size([1, C, ori_H, ori_W]), align_corners=True)
    result = F.grid_sample(img_batched, grid, align_corners=True)
    result = torch.clamp(result.squeeze(0), 0, 255)
    return result

def restore_image_cv2(box, img_tensor, ori_H, ori_W,interp=cv2.INTER_NEAREST):
    # img: (C,H,W), 可为 uint8/float32
    C, H, W = img_tensor.shape
    pts1 = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    pts2 = np.float32(box).astype(np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    # OpenCV 的 dsize 顺序是 (width, height)
    np_img = img_tensor.permute(1, 2, 0).contiguous().detach().cpu().numpy()
    warped = cv2.warpPerspective(np_img, M, (ori_W, ori_H), flags=interp,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out = torch.from_numpy(warped).permute(2, 0, 1).to(img_tensor.device, dtype=img_tensor.dtype)
    return out


def crop_image(src_img, mask):
    '''
    mask: numpy.ndarray, mask of textual, HWC
    src_img: torch.Tensor, source image, CHW
    '''
    box = min_bounding_rect(mask)
    result = adjust_image(box, src_img)
    if len(result.shape) == 2:
        result = torch.stack([result]*3, axis=-1)
    return result


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


def _check_image_file(path):
    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff'}
    return any([path.lower().endswith(e) for e in img_end])


def get_image_file_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))
    if os.path.isfile(img_file) and _check_image_file(img_file):
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if os.path.isfile(file_path) and _check_image_file(file_path):
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists


class TextRecognizer(nn.Module):
    def __init__(self, args, predictor):
        super().__init__()
        self.execution_device = torch.device("cuda")
        self.rec_image_shape = [int(v) for v in args.rec_image_shape.split(",")]
        self.rec_batch_num = args.rec_batch_num
        self.predictor = predictor
        self.chars = self.get_char_dict(args.rec_char_dict_path)
        self.char2id = {x: i for i, x in enumerate(self.chars)}
        self.is_onnx = not isinstance(self.predictor, torch.nn.Module)
        self.use_fp16 = args.use_fp16

    # img: CHW
    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[0]
        imgW = int((imgH * max_wh_ratio))

        h, w = img.shape[1:]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(imgH, resized_w),
            mode='bilinear',
            align_corners=True,
        )
        resized_image /= 255.0
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = torch.zeros((imgC, imgH, imgW), dtype=torch.float32).to(img.device)
        padding_im[:, :, 0:resized_w] = resized_image[0]
        return padding_im

    # img_list: list of tensors with shape chw 0-255
    def pred_imglist(self, img_list, show_debug=False):
        img_num = len(img_list)
        assert img_num > 0
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[2] / float(img.shape[1]))
        # Sorting can speed up the recognition process
        indices = torch.from_numpy(np.argsort(np.array(width_list)))
        batch_num = self.rec_batch_num
        preds_all = [None] * img_num
        preds_neck_all = [None] * img_num
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []

            imgC, imgH, imgW = self.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[1:]
                if h > w * 1.2:
                    img = img_list[indices[ino]]
                    img = torch.transpose(img, 1, 2).flip(dims=[1])
                    img_list[indices[ino]] = img
                    h, w = img.shape[1:]
                # wh_ratio = w * 1.0 / h
                # max_wh_ratio = max(max_wh_ratio, wh_ratio)  # comment to not use different ratio
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]], max_wh_ratio)
                if self.use_fp16:
                    norm_img = norm_img.half()
                norm_img = norm_img.unsqueeze(0)
                norm_img_batch.append(norm_img)
            norm_img_batch = torch.cat(norm_img_batch, dim=0)
            if show_debug:
                for i in range(len(norm_img_batch)):
                    _img = norm_img_batch[i].permute(1, 2, 0).detach().cpu().numpy()
                    _img = (_img + 0.5)*255
                    _img = _img[:, :, ::-1]
                    file_name = f'{indices[beg_img_no + i]}'
                    if os.path.exists(file_name + '.jpg'):
                        file_name += '_2'  # ori image
                    cv2.imwrite(file_name + '.jpg', _img)
            if self.is_onnx:
                input_dict = {}
                input_dict[self.predictor.get_inputs()[0].name] = norm_img_batch.detach().cpu().numpy()
                outputs = self.predictor.run(None, input_dict)
                preds = {}
                preds['ctc'] = torch.from_numpy(outputs[0])
                preds['ctc_neck'] = [torch.zeros(1)] * img_num
            else:
                preds = self.predictor(norm_img_batch)
            for rno in range(preds['ctc'].shape[0]):
                preds_all[indices[beg_img_no + rno]] = preds['ctc'][rno]
                preds_neck_all[indices[beg_img_no + rno]] = preds['ctc_neck'][rno]

        return torch.stack(preds_all, dim=0), torch.stack(preds_neck_all, dim=0)

    def get_char_dict(self, character_dict_path):
        character_str = []
        with open(character_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                character_str.append(line)
        dict_character = list(character_str)
        dict_character = ['sos'] + dict_character + [' ']  # eos is space
        return dict_character

    def get_text(self, order):
        char_list = [self.chars[text_id] for text_id in order]
        return ''.join(char_list)

    def decode(self, mat):
        text_index = mat.detach().cpu().numpy().argmax(axis=1)
        ignored_tokens = [0]
        selection = np.ones(len(text_index), dtype=bool)
        selection[1:] = text_index[1:] != text_index[:-1]
        for ignored_token in ignored_tokens:
            selection &= text_index != ignored_token
        return text_index[selection], np.where(selection)[0]

    def get_ctcloss(self, preds, gt_text, weight):
        if not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight).to(preds.device)
        ctc_loss = torch.nn.CTCLoss(reduction='none')
        log_probs = preds.log_softmax(dim=2).permute(1, 0, 2)  # NTC-->TNC
        targets = []
        target_lengths = []
        for t in gt_text:
            targets += [self.char2id.get(i, len(self.chars)-1) for i in t]
            target_lengths += [len(t)]
        targets = torch.tensor(targets).to(preds.device)
        target_lengths = torch.tensor(target_lengths).to(preds.device)
        input_lengths = torch.tensor([log_probs.shape[0]]*(log_probs.shape[1])).to(preds.device)
        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        loss = loss / input_lengths * weight
        return loss

    def forward(self,X, GT, Bbox_data,names,loss_type="l2"):
        all_ocr_loss = []
        all_ctc_loss = []
        all_edge_loss = []
        bsz = X.shape[0]      
        for batch in range(bsz):
            cropped_texts = []
            cropped_gt = []
            name = names[batch]
            bbs = Bbox_data[name+".png"]
            if len(bbs) == 0:
                continue
            # Create mask for each bounding box and crop the text regions
            for bb in bbs:
                # Convert bounding box to appropriate format if needed
                bb = np.array(bb, dtype=np.int32)

                # Create mask with the polygon
                pos = np.zeros((GT[batch].shape[1],GT[batch].shape[2],1), dtype=np.uint8) # h,w,c
                # print(pos.shape)
                cv2.fillPoly(pos, [bb], (255, 255, 255))

                # Crop the text region
                x_text = crop_image(X[batch], pos)
                gt_text = crop_image(GT[batch], pos)
                cropped_texts.append(x_text)
                cropped_gt.append(gt_text)

            x_list = cropped_texts + cropped_gt

            # Get predictions from the recognizer
            with torch.no_grad():
                preds, preds_neck = self.pred_imglist(x_list, show_debug=False)
            n_pairs = len(preds)//2
            # Calculate OCR loss
            if n_pairs > 0:
                # OCR loss calculation
                # preds_src = preds[:n_pairs]
                # preds_gt = preds[n_pairs:]
                preds_neck_x0 = preds_neck[:n_pairs]
                preds_neck_gt = preds_neck[n_pairs:]
                if loss_type == 'l1':
                    loss_ocr = (preds_neck_gt - preds_neck_x0).abs()
                elif loss_type == 'l2':
                    loss_ocr = torch.nn.functional.mse_loss(preds_neck_gt, preds_neck_x0, reduction='none')
                else:
                    raise NotImplementedError("unknown loss type '{loss_type}'")
                all_ocr_loss += [loss_ocr.mean()]
                # CTC Loss
                # all_ctc_loss += [self.get_ctcloss(preds_src, preds_gt, weight=1.0)]
            else:
                all_ocr_loss += [torch.tensor(0.0).to(self.execution_device)]
                # all_ctc_loss += [torch.tensor(0.0).to(self.execution_device)]
            del preds_neck
            del preds
            del x_list
            torch.cuda.empty_cache()
            # 如果有多个批次，计算平均损失
        if len(all_ocr_loss)==0:
            all_ocr_loss += [torch.tensor(0.0).to(self.execution_device)]
        all_ocr_loss = torch.stack(all_ocr_loss)
        # all_ctc_loss = torch.stack(all_ctc_loss)
        return all_ocr_loss.mean()# ,all_ctc_loss.mean()

def main():
    rec_model_dir = "./ocr_weights/ppv3_rec.pth"
    predictor = create_predictor(rec_model_dir)
    args = edict()
    args.rec_image_shape = "3, 48, 320"
    args.rec_char_dict_path = './ocr_weights/ppocr_keys_v1.txt'
    args.rec_batch_num = 6
    text_recognizer = TextRecognizer(args, predictor)
    image_dir = './test_imgs_cn'
    gt_text = ['韩国小馆']*14

    image_file_list = get_image_file_list(image_dir)
    valid_image_file_list = []
    img_list = []

    for image_file in image_file_list:
        img = cv2.imread(image_file)
        if img is None:
            print("error in loading image:{}".format(image_file))
            continue
        valid_image_file_list.append(image_file)
        img_list.append(torch.from_numpy(img).permute(2, 0, 1).float())
    try:
        tic = time.time()
        times = []
        for i in range(10):
            preds, _ = text_recognizer.pred_imglist(img_list)  # get text
            preds_all = preds.softmax(dim=2)
            times += [(time.time()-tic)*1000.]
            tic = time.time()
        print(times)
        print(np.mean(times[1:]) / len(preds_all))
        weight = np.ones(len(gt_text))
        loss = text_recognizer.get_ctcloss(preds, gt_text, weight)
        for i in range(len(valid_image_file_list)):
            pred = preds_all[i]
            order, idx = text_recognizer.decode(pred)
            text = text_recognizer.get_text(order)
            print(f'{valid_image_file_list[i]}: pred/gt="{text}"/"{gt_text[i]}", loss={loss[i]:.2f}')
    except Exception as E:
        print(traceback.format_exc(), E)


if __name__ == "__main__":
    main()
