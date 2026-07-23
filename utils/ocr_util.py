import os
from easydict import EasyDict as edict
import torch


import random

def get_random_char() -> str:
    """
    随机获取一个字符，可能是英文、数字或一个随机的常用中文字符。
    """
    # 随机选择字符类型：0=中文, 1=小写英文, 2=大写英文, 3=数字
    char_type = random.randint(0, 3)

    if char_type == 0:
        # 生成一个随机的常用中文字符
        # 常用汉字的 Unicode 编码范围是 0x4E00 到 0x9FFF
        return chr(random.randint(0x4E00, 0x9FFF))
    elif char_type == 1:
        # 生成一个随机的小写英文字母
        return chr(random.randint(ord('a'), ord('z')))
    elif char_type == 2:
        # 生成一个随机的大写英文字母
        return chr(random.randint(ord('A'), ord('Z')))
    else: # char_type == 3
        # 生成一个随机的数字
        return chr(random.randint(ord('0'), ord('9')))

def randomly_replace_chars_enhanced(text: str, replacement_rate: float = 0.3) -> str:
    """
    随机替换字符串中的部分字符。
    替换字符池包含英文、数字和更广泛的随机中文字符。

    :param text: 原始输入字符串。
    :param replacement_rate: 字符被替换的概率，介于 0.0 和 1.0 之间。
    :return: 经过随机字符替换后的新字符串。
    """
    if not 0.0 <= replacement_rate <= 1.0:
        raise ValueError("replacement_rate 必须在 0.0 和 1.0 之间")

    char_list = list(text)
    text_length = len(char_list)
    
    num_to_replace = int(text_length * replacement_rate)
    
    indices_to_replace = random.sample(range(text_length), num_to_replace)
    
    for index in indices_to_replace:
        # 获取一个随机生成的新字符并替换
        char_list[index] = get_random_char()
        
    return "".join(char_list)



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